import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv
from omegaconf import DictConfig


class BaseAgent(Agent, ABC):
    def __init__(self, cfg: DictConfig, logger: logging.Logger) -> None:
        load_dotenv()
        self.cfg = cfg
        self.logger = logger

        model_id = self.cfg.model.strip().lower()
        self.llm = (
            OpenAIChat(
                id=self.cfg.model,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=self.cfg.temperature,
            )
            if model_id.startswith("gpt-")
            else Gemini(
                id=self.cfg.model,
                api_key=os.getenv("GEMINI_API_KEY"),
                temperature=self.cfg.temperature,
            )
        )

        super().__init__(
            name=self.cfg.name,
            model=self.llm,
            description=self.cfg.description,
            instructions=[self.cfg.system_message],
            markdown=self.cfg.markdown,
        )

    @abstractmethod
    def run(self, input_data, **kwargs) -> Any:
        return super().run(input_data, **kwargs)
