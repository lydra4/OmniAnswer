import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Union

from deepeval.models.llms import GeminiModel, GPTModel
from deepeval.test_case.llm_test_case import LLMTestCase
from deepeval.test_case.mllm_test_case import MLLMTestCase
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
        output_data: Any,
        metric: Any,
        test_case_class: Type[Union[LLMTestCase, MLLMTestCase]] = LLMTestCase,
    ) -> float:
        if hasattr(metric, "model") and metric.model is None:
            setattr(metric, "model", self.llm)

        test_case = test_case_class(input=self.query, actual_output=output_data)
        score = metric.measure(test_case=test_case)

        metric_name = getattr(metric, "name", metric.__class__.__name__)
        score = 0.0 if score is None else round(score, 2)
        self.logger.info(f"{metric_name}: score={score}")
        return score

    @abstractmethod
    def evaluate_all(self) -> Dict[str, float]:
        pass
