import json
import logging
import os
from io import BytesIO
from typing import Dict, List

import requests
import torch
from bs4 import BeautifulSoup
from omegaconf import DictConfig
from PIL import Image
from scraperapi_sdk import ScraperAPIClient
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.text.bert import BERTScore
from torchvision.transforms.functional import pil_to_tensor


class EvaluationPipeline:
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.result_dict = self._load_results_dict(path=self.cfg.output_path)
        modes: List[str] = [
            result_dict["modality"] for result_dict in self.result_dict["results"]
        ]
        if "text" in modes:
            self.text_scrap_client: ScraperAPIClient = ScraperAPIClient(
                api_key=os.getenv("SCRAPER_API_KEY")
            )
            self.text_metric: BERTScore = BERTScore(
                model_name_or_path=self.cfg.text.text_model_name,
                device=self.device,
            )

        if "image" in modes:
            self.image_metrics = CLIPScore(
                model_name_or_path=self.cfg.image.image_model_name,
            )

        if "video" in modes:
            pass

    def _load_results_dict(self, path: str) -> Dict[str, str]:
        result_dict_path = os.path.join(path, "evaluation_dict.json")
        if os.path.exists(result_dict_path):
            try:
                with open(result_dict_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error occured: {e}.")

    def _scrap_text(self, url: str, num_words: int) -> str:
        self.logger.info(f"Web scraping '{url}'.")
        response = self.text_scrap_client.get(url=url, params={"render": True})
        soup = BeautifulSoup(response, "html.parser")
        article = soup.find("article")
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

    def evaluate(self):
        original_query = self.result_dict["query"]
        self.logger.info(f"Evaluating query:'{original_query}'.")
        self._evaluate_text(num_words=self.cfg.text.num_words)
        self._evaluate_image()
