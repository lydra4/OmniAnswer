import ast
import logging
from typing import Any, List, Optional

import nltk
from omegaconf import DictConfig

from agents.base_agent import BaseAgent

nltk.download("punkt")


class ModalityAgent(BaseAgent):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
        tools: Optional[List[Any]] = None,
    ) -> None:
        tools = [] if tools is None else tools
        super().__init__(
            cfg=cfg.modality_agent,
            logger=logger,
            llm=llm,
            tools=tools,
        )

    def run_query(self, query: str, **kwargs) -> List[str]:
        self._logger.info(f"Running on query: '{query}'.")
        result = self._llm.call(query, **kwargs)
        self._logger.info(
            f"For the query:'{query}', best modes of learning: '{result}'."
        )
        result_cleaned = ast.literal_eval(result.__dict__["raw"])
        return result_cleaned
