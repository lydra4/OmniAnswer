import ast
import logging
import re
from typing import Dict, List

from agents.base_agent import BaseAgent
from omegaconf import DictConfig


class ParaphraseAgent(BaseAgent):
    def __init__(self, cfg: DictConfig, logger: logging.Logger, llm) -> None:
        super().__init__(cfg=cfg.paraphrase_agent, logger=logger, llm=llm)

    def run(self, query: str, modalities: List[str], **kwargs) -> Dict[str, str]:
        """
        Run the paraphrase agent to generate paraphrases for the given query.

        Args:
            query (str): The input query to be paraphrased.
            modalities (List[str]): List of modalities to consider for paraphrasing.

        Returns:
            Dict[str, str]: A dictionary containing the paraphrased query.
        """
        self.logger.info(f"Running ParaphraseAgent with query: {query}.")
        response = super().run(query, modalities=modalities, **kwargs)
        cleaned = re.sub(
            r"^```(?:json|python)?\s*|\s*```$",
            "",
            response.content.strip(),
            flags=re.IGNORECASE,
        )
        paraphrased_outputs = ast.literal_eval(cleaned)

        return paraphrased_outputs
