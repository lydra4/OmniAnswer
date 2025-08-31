import logging
import os
from typing import Any, List, Optional

import nltk
from omegaconf import DictConfig
from openai import OpenAI

from agents.base_agent import BaseAgent
from utils.general_utils import extract_python_json_block

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
        self._logger.info(f'Running on query: "{query}".')
        response = super().run_query(query=query)
        modalities = extract_python_json_block(response.content.strip())
        self._logger.info(
            f'For the query:"{query}", best modes of learning: "{modalities}".'
        )
        return modalities
