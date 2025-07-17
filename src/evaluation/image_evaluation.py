import logging
from io import BytesIO
from typing import Dict

import requests
from evaluation.base_evaluation import BaseEvaluation
from omegaconf import DictConfig
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ImageEvaluation(BaseEvaluation):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        query: str,
        output: Dict[str, str],
    ):
        super().__init__(cfg=cfg, logger=logger, query=query, output=output)
        self.model = CLIPModel.from_pretrained(
            pretrained_model_name_or_path=self.cfg.image_evaluator.pretrained_model_name_or_path
        )
        self.processor = CLIPProcessor.from_pretrained(
            pretrained_model_name_or_path=self.cfg.image_evaluator.pretrained_model_name_or_path
        )

    def _load_image(self) -> Image.Image:
        response = requests.get(self.output["image"])
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    def _evaluate_clip_relevancy(self, image: Image.Image) -> float:
        inputs = self.processor(
            text=[self.query], images=image, return_tensors="pt", padding=True
        )
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        return round(probs[0][1].item(), 2)
