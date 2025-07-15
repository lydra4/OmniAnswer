import logging
import os
from typing import Dict

from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.models.llms import GeminiModel, GPTModel
from deepeval.test_case.llm_test_case import LLMTestCase, LLMTestCaseParams
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

    def evaluate_text_agent(self):
        self.logger.info("Evaluating relevancy on text agent.")
        relevancy_metric = AnswerRelevancyMetric(
            model=self.llm,
            include_reason=False,
        )
        test_case = LLMTestCase(input=self.query, actual_output=self.output["text"])

        relevancy_metric.measure(test_case, _show_indicator=False)
        self.logger.info(f"Relevancy score: {relevancy_metric.score}.")

    def evaluate_with_llm(self):
        test_case = LLMTestCase(input=self.query, actual_output=self.output["text"])

        factuality = GEval(
            name=self.cfg.text_evaluator.factuality_name,
            criteria=self.cfg.text_evaluator.factuality_criteria,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
        )

        clarity = GEval(
            name=self.cfg.text_evaluator.clarity_name,
            criteria=self.cfg.text_evaluator.clarity_criteria,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
        )

        conciseness = GEval(
            name=self.cfg.text_evaluator.conciseness_name,
            criteria=self.cfg.text_evaluator.conciseness_criteria,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
        )

        factuality.measure(test_case=test_case, _show_indicator=False)
        clarity.measure(test_case=test_case, _show_indicator=False)
        conciseness.measure(test_case=test_case, _show_indicator=False)
        print(
            factuality.score,
            clarity.score,
            conciseness.score,
        )
