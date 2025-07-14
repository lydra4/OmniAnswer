import logging
import os
from typing import Dict

from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from deepeval.models.llms import GeminiModel, GPTModel
from deepeval.test_case.llm_test_case import LLMTestCase
from omegaconf import DictConfig


class EvaluationPipeline:
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        query: str,
        output: Dict[str, str],
    ):
        """
        Initializes the evaluation pipeline with configuration and logger.

        Args:
            cfg (DictConfig): Configuration for the evaluation pipeline.
            logger (logging.Logger): Logger instance for tracking execution.
        """
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

    def evaluate_text_agent(self, text_output: str):
        relevancy_metric = AnswerRelevancyMetric(model=self.llm, include_reason=False)
        test_case = LLMTestCase(input=self.query, actual_output=self.output["text"])

        relevancy_metric.measure(test_case)
        print(relevancy_metric.score)
