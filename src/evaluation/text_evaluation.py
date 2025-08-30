import logging
from typing import Dict, Union

from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.test_case.llm_test_case import LLMTestCase, LLMTestCaseParams
from omegaconf import DictConfig

from evaluation.base_evaluation import BaseEvaluation


class TextEvaluation(BaseEvaluation):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
    ):
        super().__init__(
            cfg=cfg,
            logger=logger,
        )

    def _evaluate_relevancy(
        self,
        query: str,
        output_text: str,
    ) -> float:
        metric = AnswerRelevancyMetric(model=self.llm, include_reason=False)
        return self._execute_deepeval_metric(
            query=query,
            output_data=output_text,
            metric=metric,
            test_case_class=LLMTestCase,
        )

    def _evaluate_llm_metrics(
        self,
        query: str,
        output_text: str,
    ) -> Dict[str, float]:
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
            scores[metric_key] = self._execute_deepeval_metric(
                query=query,
                output_data=output_text,
                metric=metric,
                test_case_class=LLMTestCase,
            )

        return scores

    def evaluate_all(
        self,
        query: str,
        output_data: Dict[str, Union[str, Dict[str, str]]],
    ):
        self.logger.info(f"Evaluating text agent on query:{query}.")
        output_text = output_data["results"]["text"]

        scores = {
            "relevancy": self._evaluate_relevancy(query=query, output_text=output_text),
            **self._evaluate_llm_metrics(query=query, output_text=output_text),
        }
        self.logger.info(f"Text agent scores: {scores}.")
        return scores
