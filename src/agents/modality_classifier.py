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


class ModalityClassifier(BaseAgent):
    def __init__(self, cfg: DictConfig, logger: logging.Logger):
        super().__init__(cfg=cfg, logger=logger)
        self.guard = Guard().use_many(
            BanList(
                banned_words=[
                    "meth",
                    "rape",
                    "murder",
                    "porn",
                    "suicide",
                    "drug",
                    "sex",
                    "kill",
                ],
                on_fail="refrain",
            ),
            ToxicLanguage(
                threshold=0.5,
                validation_method="sentence",
                on_fail="refrain",
            ),
        )

    def run(self, query: str) -> List[str]:
        result = self.guard.validate(query)
        if not result.validation_passed:
            self.logger.warning(f"Rejected query: {result.error}")
            raise ValueError(f"Query rejected: {result.error}")

        response = super().run(query)
        cleaned = re.sub(
            r"^```(?:json|python)?\s*|\s*```$",
            "",
            response.content.strip(),
            flags=re.IGNORECASE,
        )
        modalities = ast.literal_eval(cleaned)
        return modalities
