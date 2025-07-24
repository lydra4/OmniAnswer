import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Union

from deepeval.models.llms.gemini_model import GeminiModel
from deepeval.models.llms.openai_model import GPTModel
from deepeval.test_case.llm_test_case import LLMTestCase
from deepeval.test_case.mllm_test_case import MLLMTestCase
from dotenv import load_dotenv
from omegaconf import DictConfig


class BaseEvaluation(ABC):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
    ):
        load_dotenv()
        self.cfg = cfg
        self.logger = logger
        self.llm = (
            GPTModel(model=self.cfg.model, _openai_api_key=os.getenv("OPENAI_API_KEY"))
            if "gpt-" in self.cfg.model.lower()
            else GeminiModel(
                model_name=self.cfg.model, api_key=os.getenv("GEMINI_API_KEY")
            )
        )

    def _execute_deepeval_metric(
        self,
        query: str,
        output_data: str,
        metric: Any,
        test_case_class: Type[Union[LLMTestCase, MLLMTestCase]] = LLMTestCase,
    ) -> float:
        if hasattr(metric, "model") and metric.model is None:
            setattr(metric, "model", self.llm)

        test_case = test_case_class(input=query, actual_output=output_data)
        score = metric.measure(test_case=test_case)

        metric_name = getattr(metric, "name", metric.__class__.__name__)
        score = 0.0 if score is None else round(score, 2)
        self.logger.info(f"{metric_name}: score={score}")
        return score

    @abstractmethod
    def evaluate_all(
        self,
        query: str,
        output_data: Dict[str, Union[str, Dict[str, str]]],
    ) -> Dict[str, float]:
        pass
