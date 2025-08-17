import logging
import os
from typing import Any, List, Optional

import nltk
from agents.base_agent import BaseAgent
from omegaconf import DictConfig
from openai import OpenAI
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
        super().__init__(cfg=cfg.modality_agent, logger=logger, llm=llm, tools=tools)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def run_query(self, query: str, **kwargs) -> List[str]:
        self.logger.info(f'Running on query: "{query}".')
        [response] = self.client.moderations.create(
            model=self.cfg.moderation_model, input=query
        ).model_dump()["results"]
        true_categories: List[str] = [
            key for key, value in response["categories"].items() if value is True
        ]

        if true_categories:
            self.logger.error(
                f"Rejected query due to: {' and '.join(true_categories)}."
            )
            raise ValueError(f"Rejected query due to: {'and '.join(true_categories)}.")

        response = super().run(message=query)
        modalities = extract_python_json_block(response.content.strip())
        self.logger.info(
            f'For the query:"{query}", best modes of learning: "{modalities}".'
        )
        return modalities
