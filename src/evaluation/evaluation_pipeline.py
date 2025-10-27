import json
import logging
import os
import subprocess
from io import BytesIO
from typing import List

import av
import requests
import torch
from bs4 import BeautifulSoup
from omegaconf import DictConfig
from PIL import Image
from schemas.schemas import ResultDictFile
from scraperapi_sdk import ScraperAPIClient
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.text.bert import BERTScore
from torchvision.transforms.functional import pil_to_tensor
from transformers import XCLIPModel, XCLIPProcessor
from yt_dlp import YoutubeDL


class EvaluationPipeline:
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.result_dict: ResultDictFile = self._load_results_dict(
            path=self.cfg.output_path
        )
        modes: List[str] = [
            result_dict["modality"] for result_dict in self.result_dict["results"]
        ]
        if "text" in modes:
            api_key = os.getenv("SCRAPER_API_KEY")
            if api_key is None:
                raise ValueError("Scraper API Key must be set")
            self.text_scrap_client: ScraperAPIClient = ScraperAPIClient(api_key=api_key)
            self.text_metric: BERTScore = BERTScore(
                model_name_or_path=self.cfg.text.text_model_name,
                device=self.device,
            )

        if "image" in modes:
            self.image_metrics = CLIPScore(
                model_name_or_path=self.cfg.image.image_model_name,
            )

        if "video" in modes:
            self.video_processor = XCLIPProcessor.from_pretrained(
                pretrained_model_name_or_path=self.cfg.video.video_model_name,
                use_fast=True,
            )
            self.video_model = XCLIPModel.from_pretrained(
                self.cfg.video.video_model_name
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
        if article is None:
            self.logger.warning(f"No article tag found at {url}, return empty text.")
            return ""
        paragraphs = [p.get_text(strip=True) for p in article.find_all("p")]
        text_content = " ".join(paragraphs)
        self.logger.info("Web scrap successfull.")
        return " ".join(text_content.split()[:num_words])

    def _evaluate_text(self, num_words: int) -> None:
        url, query = next(
            (result_dict["url"], result_dict["paraphrase"])
            for result_dict in self.result_dict["results"]
            if result_dict["modality"] == "text"
        )
        text_content = self._scrap_text(url=url, num_words=num_words)
        bert_score = self.text_metric([query], [text_content])
        precision = bert_score["precision"]
        self.logger.info(f"The text score is {precision:.2f}.")

    def _scrap_image(self, url: str) -> torch.Tensor:
        self.logger.info(f"Web scraping '{url}'.")
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

    def _evaluate_image(self) -> None:
        url, query = next(
            (result_dict["url"], result_dict["paraphrase"])
            for result_dict in self.result_dict["results"]
            if result_dict["modality"] == "image"
        )
        image_tensor = self._scrap_image(url=url)
        raw_score = self.image_metrics(image_tensor, query)
        score = raw_score.detach().item()
        self.logger.info(f"The image score is {score:.2f}.")

    def _get_stream_url(self, url: str) -> str:
        ydl_opts = {
            "noplaylist": True,
            "quiet": True,
            "cookiefile": "./data/cookies/all_cookies.txt",
            "format": "bestvideo[height<=720]/bestvideo/best",
            "retries": 3,
            "forceipv4": True,
            "geo_bypass": True,
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
            img = frame.to_ndarray(format="rgb24")
            frames.append(img)
        return [Image.fromarray(f.astype("uint8"), mode="RGB") for f in frames]

    def _evaluate_video(self, duration: int):
        url, query = next(
            (result_dict["url"], result_dict["paraphrase"])
            for result_dict in self.result_dict["results"]
            if result_dict["modality"] == "video"
        )
        stream_url = self._get_stream_url(url=url)
        frames = self._load_video_to_ram(stream_url=stream_url, duration=duration)
        inputs = self.video_processor(
            text=[query],
            images=frames,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.video_model(**inputs)
            video_embeds = outputs.video_embeds
            text_embeds = outputs.text_embeds

            similarity = torch.nn.functional.cosine_similarity(
                video_embeds, text_embeds
            )
        print(similarity)

    def evaluate(self):
        original_query = self.result_dict["query"]
        self.logger.info(f"Evaluating query:'{original_query}'.")
        # self._evaluate_text(num_words=self.cfg.text.num_words)
        # self._evaluate_image()
        self._evaluate_video(duration=self.cfg.video.duration)
