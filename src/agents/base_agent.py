import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from agno.agent import Agent
from dotenv import load_dotenv
from omegaconf import DictConfig


class BaseAgent(Agent, ABC):
    def __init__(self, cfg: DictConfig, logger: logging.Logger, llm) -> None:
        load_dotenv()
        self.cfg = cfg
        self.logger = logger
        self.llm = llm

        os.environ["AGNO_API_KEY"] = os.getenv("AGNO_API_KEY")
        os.environ["AGNO_MONITOR"] = os.getenv("AGNO_MONITOR")

        super().__init__(
            name=self.cfg.name,
            model=self.llm,
            description=self.cfg.description,
            instructions=[self.cfg.system_message],
            markdown=True,
            monitoring=True,
            show_tool_calls=True,
        )

    @abstractmethod
    def run(self, input_data, **kwargs) -> Any:
        return super().run(input_data, **kwargs)
