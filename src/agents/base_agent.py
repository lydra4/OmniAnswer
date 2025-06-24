import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import openlit
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv
from langfuse import get_client
from omegaconf import DictConfig


class BaseAgent(Agent, ABC):
    def __init__(self, cfg: DictConfig, logger: logging.Logger) -> None:
        load_dotenv()
        self.cfg = cfg
        self.logger = logger
        os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
        os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
        os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")

        self.langfuse = get_client()
        self.langfuse.auth_check()
        openlit.init(
            tracer=self.langfuse._otel_tracer,
            disable_batch=True,
        )

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
