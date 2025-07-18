import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

from deepeval.metrics.base_metric import BaseMetric
from deepeval.models.llms import GeminiModel, GPTModel
from deepeval.test_case.llm_test_case import LLMTestCase
from omegaconf import DictConfig


class BaseEvaluation(ABC):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        query: str,
        output: Dict[str, str],
    ):
        self.cfg = cfg
        self.logger = logger
        self.query = query
        self.output = output
        self.llm = (
            GPTModel(model=self.cfg.model, api_key=os.getenv("OPENAI_API_KEY"))
            if self.cfg.model.startswith("gpt-")
            else GeminiModel(
                model_name=self.cfg.model, api_key=os.getenv("GEMINI_API_KEY")
            )
        )

    def _execute_deepeval_metric(
        self,
        input_data: str,
        output_data: Any,
        metric: BaseMetric,
        test_case_class=LLMTestCase,
    ):
        if hasattr(metric, "model") and metric.model is None:
            metric.model = self.llm

        test_case = test_case_class(input=input_data, actual_output=output_data)
        score = metric.measure(test_case=test_case)
        return round(score, 2)

    @abstractmethod
    def evaluate_all(self) -> Dict[str, float]:
        pass
