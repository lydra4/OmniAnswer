import ast
import logging
from typing import Dict, List

from agents.base_agent import BaseAgent
from omegaconf import DictConfig


class ParaphraseAgent(BaseAgent):
    def __init__(self, cfg: DictConfig, logger: logging.Logger):
        super().__init__(cfg=cfg, logger=logger)
        self.cfg = cfg
        self.logger = logger

    def run(self, query: str, modalities: List[str]) -> Dict[str, str]:
        """
        Run the paraphrase agent to generate paraphrases for the given query.

        Args:
            query (str): The input query to be paraphrased.
            modalities (List[str]): List of modalities to consider for paraphrasing.

        Returns:
            Dict[str, str]: A dictionary containing the paraphrased query.
        """
        self.logger.info(f"Running ParaphraseAgent with query: {query}.")
        response = super().run(query, modalities)
        modalities = ast.literal_eval(response.content.strip())

        return modalities
