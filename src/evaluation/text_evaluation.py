import logging
import os
from typing import Dict

from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.models.llms import GeminiModel, GPTModel
from deepeval.test_case.llm_test_case import LLMTestCase, LLMTestCaseParams
from omegaconf import DictConfig


class TextEvaluation:
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

    def _evaluate_relevancy(self) -> float:
        test_case = LLMTestCase(input=self.query, actual_output=self.output["text"])
        metric = AnswerRelevancyMetric(model=self.llm, include_reason=False)
        metric.measure(test_case, _show_indicator=False)
        return round(metric.score if metric.score is not None else 0.0, 2)

    def _evaluate_llm_metrics(self):
        test_case = LLMTestCase(input=self.query, actual_output=self.output["text"])

        metrics_config = {
            "factuality": {
                "name": self.cfg.text_evaluator.factuality_name,
                "criteria": self.cfg.text_evaluator.factuality_criteria,
            },
            "clarity": {
                "name": self.cfg.text_evaluator.clarity_name,
                "criteria": self.cfg.text_evaluator.clarity_criteria,
            },
            "conciseness": {
                "name": self.cfg.text_evaluator.conciseness_name,
                "criteria": self.cfg.text_evaluator.conciseness_criteria,
            },
        }

        scores = {}
        for metric_key, metric_info in metrics_config.items():
            metric = GEval(
                name=metric_info["name"],
                criteria=metric_info["criteria"],
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                model=self.llm,
            )
            metric.measure(test_case, _show_indicator=False)
            scores[metric_key.capitalize()] = round(
                metric.score if metric.score is not None else 0.0, 2
            )

        return scores

    def evaluate_all(self) -> Dict[str, float]:
        self.logger.info("Evaluating text agent.")
        relevancy = self._evaluate_relevancy()
        llm_scores = self._evaluate_llm_metrics()

        scores = {"relevancy": relevancy, **llm_scores}
        self.logger.info(f"Text agent scores: {scores}.")
        return scores
