import ast
import logging
import os
from typing import List

import omegaconf
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv


class ModalityClassifier(Agent):
    def __init__(self, cfg: omegaconf.DictConfig, logger: logging.Logger) -> None:
        load_dotenv()
        self.cfg = cfg
        self.logger = logger
        self.llm = (
            OpenAIChat(
                id=self.cfg.model,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=self.cfg.temperature,
            )
            if "gpt-" in self.cfg.model.strip().lower()
            else Gemini(
                id=self.cfg.model,
                api_key=os.getenv("GEMINI_API_KEY"),
                temperature=self.cfg.temperature,
            )
        )

        super().__init__(
            name=cfg.name,
            model=self.llm,
            description=cfg.description,
            instructions=[cfg.system_message],
            markdown=cfg.markdown,
        )

    def run(self, query: str) -> List[str]:
        response = super().run(query)
        modalities = ast.literal_eval(response.content)
        return modalities
