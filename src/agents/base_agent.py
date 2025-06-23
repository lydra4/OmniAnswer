import logging
import os
from abc import ABC
from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv
from guardrails import Guard
from omegaconf import DictConfig


class BaseAgent(Agent, ABC):
    def __init__(self, cfg: DictConfig, logger: logging.Logger) -> None:
        load_dotenv()
        self.cfg = cfg
        self.logger = logger
        self.guard = Guard(validators=[SafeText(threshold="low")])

        model_id = self.cfg.model.strip().lower()
        self.llm = (
            OpenAIChat(
                id=self.cfg.model,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=self.cfg.temperature,
            )
            if "gpt-" in model_id
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

    def run(self, query: str) -> Any:
        response = super().run(query)
        validated = self.guard.validate(response.content)
        if not validated["pass"]:
            self.logger.warning("Block unsafe or restricted content.")
            raise ValueError("Response blocked due to unsafe or off-topic content.")

        return response.content
