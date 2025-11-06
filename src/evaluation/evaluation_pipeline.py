import json
import logging
import os
import subprocess
from io import BytesIO
from typing import List, Optional, Tuple

import av
import numpy as np
import requests
import torch
from bs4 import BeautifulSoup
from omegaconf import DictConfig
from PIL import Image
from scraperapi_sdk import ScraperAPIClient
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.text.bert import BERTScore
from torchvision.transforms.functional import pil_to_tensor
from transformers import XCLIPModel, XCLIPProcessor
from yt_dlp import YoutubeDL

from schemas.schemas import ResultDictFile
from utils.general_utils import init_mlflow


class EvaluationPipeline:
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        result_dict: ResultDictFile,
        llm_name: str,
        temperature: int | float,
    ) -> None:
        self.cfg = cfg.evaluation
        self.logger = logger
        self.result_dict = result_dict
        self.llm_name = llm_name
        self.temperature = temperature
        self.mlflow_directory: str = self.cfg.mlflow_directory
        self.modes: List[str] = [
            result_dict["modality"] for result_dict in self.result_dict["results"]
        ]
        self.experiment_name: str = getattr(
            self.cfg, "experiment_name", "evaluation_ablation"
        )

        if "text" in self.modes:
            api_key = os.getenv("SCRAPER_API_KEY")
            if api_key is None:
                raise ValueError("Scraper API Key must be set")
            self.text_scrap_client: ScraperAPIClient = ScraperAPIClient(api_key=api_key)
            self.text_metric: BERTScore = BERTScore(
                model_name_or_path=self.cfg.text.text_model_name
            )

        if "image" in self.modes:
            self.image_metrics = CLIPScore(
                model_name_or_path=self.cfg.image.image_model_name,
            )

        if "video" in self.modes:
            self.video_processor = XCLIPProcessor.from_pretrained(
                pretrained_model_name_or_path=self.cfg.video.video_model_name,
                use_fast=True,
            )
            self.video_model = XCLIPModel.from_pretrained(
                self.cfg.video.video_model_name
            )
            self.clip_length = getattr(
                self.video_model.config.vision_config, "num_frames", None
            )

    def _load_results_dict(self, path: str) -> ResultDictFile:
        result_dict_path = os.path.join(path, "evaluation_dict.json")
        if os.path.exists(result_dict_path):
            try:
                with open(result_dict_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error occured: {e}.")
                raise
        return {"query": "", "results": []}

    def _scrap_text(self, url: str, num_words: int) -> str:
        self.logger.info(f"Web scraping url:'{url}'.")
        response = self.text_scrap_client.get(url=url, params={"render": True})
        soup = BeautifulSoup(response, "html.parser")
        article = soup.find("article")
        if article:
            text = " ".join(p.get_text(strip=True) for p in article.find_all("p"))
        else:
            self.logger.warning(
                f"No <article> tag found at {url}, falling back to <body>."
            )
            body = soup.find("body")
            text = body.get_text(separator=" ", strip=True) if body else ""

        clean_text = " ".join(text.split()[:num_words])
        self.logger.info("Web scrap successful.")
        return clean_text

    def _evaluate_text(self, num_words: int) -> Optional[float]:
        url, query = next(
            (
                (result_dict["url"], result_dict["paraphrase"])
                for result_dict in self.result_dict["results"]
                if result_dict["modality"] == "text"
            ),
            (None, None),
        )
        if url is None:
            return None
        text_content = self._scrap_text(url=url, num_words=num_words)
        bert_score = self.text_metric([query], [text_content])
        precision = bert_score["precision"]
        self.logger.info(f"The text score is {precision:.2f}.")
        return precision

    def _scrap_image(self, url: str) -> torch.Tensor:
        self.logger.info(f"Image scraping '{url}'.")
        try:
            response = requests.get(url=url, timeout=10)
            response.raise_for_status()
            with BytesIO(response.content) as image_memory:
                image = Image.open(image_memory).convert("RGB")
            self.logger.info("Web scrap successfull.")
            image_tensor = pil_to_tensor(pic=image)
            return image_tensor.unsqueeze(0)
        except Exception as e:
            raise ValueError(f"Error occurred: {e}.") from e

    def _evaluate_image(self) -> Optional[float]:
        url, query = next(
            (
                (result_dict["url"], result_dict["paraphrase"])
                for result_dict in self.result_dict["results"]
                if result_dict["modality"] == "image"
            ),
            (None, None),
        )
        if url is None:
            return None
        image_tensor = self._scrap_image(url=url)
        raw_score = self.image_metrics(image_tensor, query)
        score = raw_score.detach().item()
        self.logger.info(f"The image score is {score:.2f}.")
        return score

    def _get_stream_url(self, url: str) -> str:
        ydl_opts = {
            "noplaylist": True,
            "quiet": True,
            "cookiefile": "./data/cookies/all_cookies.txt",
            "format": "bestvideo[height<=720]/bestvideo/best",
            "retries": 3,
            "forceipv4": True,
            "nocheckcertificate": True,
            "headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                ),
            },
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url=url, download=False)
        formats = info["formats"]
        video_formats = [
            f for f in formats if f.get("vcodec") and f["vcodec"] != "none"
        ]
        if not video_formats:
            raise RuntimeError("No video formats found")

        best = max(video_formats, key=lambda f: f.get("height") or 0)
        return best["url"]

    def _load_video_to_ram(self, stream_url: str, duration: int) -> List[Image.Image]:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            stream_url,
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-f",
            "mpegts",
            "pipe:1",
        ]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_data, stderr_data = proc.communicate(timeout=60 + duration)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed: {stderr_data.decode('utf8', errors='ignore')}"
            )

        video_buffer = BytesIO(stdout_data)
        container = av.open(video_buffer, format="mpegts")
        frames = []
        for frame in container.decode(video=0):
            pts_seconds = float(frame.pts or 0) * float(frame.time_base or 0)
            if pts_seconds > duration:
                break
            img = Image.fromarray(frame.to_ndarray(format="rgb24"))
            frames.append(img)
        return frames

    def _sample_frame_indices(
        self,
        video: List[Image.Image],
        clip_length: int,
        segment_length: int,
    ) -> List[Image.Image]:
        indices = np.linspace(
            start=0, stop=(segment_length - 1), num=clip_length
        ).astype(np.int64)
        index_frames = [video[i] for i in indices]
        return index_frames

    def _preprocess_video_text(
        self, video: List[Image.Image], query: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        procesed_video = self.video_processor.video_processor.preprocess(
            video, return_tensors="pt"
        )
        processed_text = self.video_processor.tokenizer(
            [query], return_tensors="pt", padding=True
        )

        inputs = {
            "pixel_values": procesed_video["pixel_values"],
            "input_ids": processed_text["input_ids"],
            "attention_mask": processed_text["attention_mask"],
        }
        with torch.no_grad():
            outputs = self.video_model(**inputs)

        text_embeds = outputs["text_embeds"].squeeze(1)
        video_embeds = outputs["video_embeds"]
        return text_embeds, video_embeds

    def _evaluate_video(self, duration: int) -> Optional[float]:
        url, query = next(
            (
                (result_dict["url"], result_dict["paraphrase"])
                for result_dict in self.result_dict["results"]
                if result_dict["modality"] == "video"
            ),
            (None, None),
        )
        if url is None:
            return None
        stream_url = self._get_stream_url(url=url)
        video = self._load_video_to_ram(stream_url=stream_url, duration=duration)
        index_frames = self._sample_frame_indices(
            video=video,
            clip_length=self.clip_length,
            segment_length=len(video),
        )
        text_embeds, video_embeds = self._preprocess_video_text(
            video=index_frames, query=query
        )

        text_embeds_norm = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)
        video_embeds_norm = torch.nn.functional.normalize(video_embeds, p=2, dim=-1)
        sim = torch.nn.functional.cosine_similarity(
            text_embeds_norm, video_embeds_norm, dim=-1
        ).item()

        self.logger.info(f"The video score is {sim:.2f}.")
        return sim

    def evaluate(self):
        original_query = self.result_dict["query"]
        self.logger.info(f"Evaluating query:'{original_query}'.")

        text_sim = self._evaluate_text(num_words=self.cfg.text.num_words)
        image_sim = self._evaluate_image()
        video_sim = self._evaluate_video(duration=self.cfg.video.duration)

        init_mlflow(
            directory=self.mlflow_directory,
            experiment_name=self.experiment_name,
            llm_name=self.llm_name,
            temperature=self.temperature,
            modes=self.modes,
            text_similarity=text_sim,
            image_similarity=image_sim,
            video_similarity=video_sim,
        )
