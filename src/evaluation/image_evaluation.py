import logging
from io import BytesIO
from typing import Dict

import requests
from deepeval import evaluate
from deepeval.metrics.multimodal_metrics.image_coherence.image_coherence import (
    ImageCoherenceMetric,
)
from deepeval.test_case.mllm_test_case import MLLMImage, MLLMTestCase
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

    def _load_image(self, url: str) -> Image.Image:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    def _evaluate_clip_relevancy(self, image: Image.Image) -> float:
        inputs = self.processor(
            text=[self.query], images=image, return_tensors="pt", padding=True
        )
        output = self.model(**inputs).logits_per_image.item()
        return round(output, 2)

    def _image_coherence(self, url: str) -> float:
        metric = ImageCoherenceMetric()
        test_case = MLLMTestCase(
            input=[self.query],
            actual_output=[MLLMImage(url=self.output["image"], local=False)],
        )
        evaluate(test_cases=[test_case], metrics=[metric])
        return round(metric.measure(test_case=test_case), 2)
