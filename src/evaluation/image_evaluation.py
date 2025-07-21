import logging
from io import BytesIO
from typing import Dict

import requests
from deepeval.metrics.multimodal_metrics.image_coherence.image_coherence import (
    ImageCoherenceMetric,
)
from deepeval.metrics.multimodal_metrics.multimodal_g_eval.multimodal_g_eval import (
    MultimodalGEval,
)
from deepeval.test_case.mllm_test_case import (
    MLLMImage,
    MLLMTestCase,
    MLLMTestCaseParams,
)
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
        return self._execute_deepeval_metric(
            output_data=[MLLMImage(url=url, local=False)],
            metric=metric,
            test_case_class=MLLMTestCase,
        )

    def _evaluate_llm_metrics(self, url: str) -> Dict[str, float]:
        metrics_config = {
            "Instruction Compliance": {
                "name": self.cfg.image_evaluator.instruction_compliance_name,
                "criteria": self.cfg.image_evaluator.instruction_compliance_criteria,
            },
            "Factual alignment": {
                "name": self.cfg.image_evaluator.factual_alignment_name,
                "criteria": self.cfg.image_evaluator.factual_alignment_criteria,
            },
            "Aesthetics": {
                "name": self.cfg.image_evaluator.aesthetics_name,
                "criteria": self.cfg.image_evaluator.aesthetics_criteria,
            },
        }

        scores = {}
        for metric_key, metric_info in metrics_config.items():
            metric = MultimodalGEval(
                name=metric_info["name"],
                criteria=metric_info["criteria"],
                evaluation_params=[
                    MLLMTestCaseParams.INPUT,
                    MLLMTestCaseParams.ACTUAL_OUTPUT,
                ],
            )
            scores[metric_key] = self._execute_deepeval_metric(
                output_data=url,
                metric=metric,
                test_case_class=MLLMTestCase,
            )

        return scores

    def evaluate_all(self) -> Dict[str, float]:
        self.logger.info("Evaluating image agent")

        image = self._load_image(url=self.output["image"])
        scores = {
            "relevancy": self._evaluate_clip_relevancy(image=image),
            "coherence": self._image_coherence(url=self.output["image"]),
            **self._evaluate_llm_metrics(url=self.output["image"]),
        }

        self.logger.info(f"Image agent scores: {scores}")
        return scores
