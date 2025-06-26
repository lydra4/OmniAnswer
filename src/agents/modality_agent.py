import ast
import logging
import re
from typing import List

import nltk
from agents.base_agent import BaseAgent
from guardrails.guard import Guard
from guardrails.hub import BanList, ToxicLanguage
from omegaconf import DictConfig

nltk.download("punkt")


class ModalityAgent(BaseAgent):
    def __init__(self, cfg: DictConfig, logger: logging.Logger):
        super().__init__(cfg=cfg.modality_agent, logger=logger)
        self.guard = Guard().use_many(
            BanList(
                banned_words=cfg.modality_agent.guardrails.banned_words,
                on_fail="refrain",
            ),
            ToxicLanguage(
                threshold=cfg.modality_agent.guardrails.toxic_threshold,
                validation_method="sentence",
                on_fail="refrain",
            ),
        )

    def run(self, query: str) -> List[str]:
        """
        Run the ModalityAgent to determine the modalities of the given query.

        Args:
            query (str): The input query to analyze for modalities.

        Returns:
            List[str]: A list of modalities identified in the query.
        """
        self.logger.info(f"Running ModalityAgent with query: {query}.")
        result = self.guard.validate(query)

        if not result.validation_passed:
            reasons_by_validator = {}

            for summary in result.validation_summaries:
                if summary.validator_status == "fail":
                    all_reasons = [span.reason for span in summary.error_spans]
                    reasons_by_validator[summary.validator_name] = all_reasons

                    for reason in all_reasons:
                        self.logger.warning(
                            f"{summary.validator_name} failed: {reason}"
                        )

            formatted = []
            for validator, reasons in reasons_by_validator.items():
                for reason in reasons:
                    formatted.append(f"{validator}: {reason}")

            raise ValueError("Rejected query due to:\n" + "\n".join(formatted))

        response = super().run(query)
        cleaned = re.sub(
            r"^```(?:json|python)?\s*|\s*```$",
            "",
            response.content.strip(),
            flags=re.IGNORECASE,
        )
        modalities = ast.literal_eval(cleaned)
        return modalities
