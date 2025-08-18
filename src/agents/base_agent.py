import logging
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from agno.agent import Agent
from omegaconf import DictConfig


class BaseAgent(Agent, ABC):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
        tools: Optional[List[Any]] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.llm = llm

        agno_api_key = os.getenv("AGNO_API_KEY")
        agno_monitor = os.getenv("AGNO_MONITOR")

        if agno_api_key is not None:
            os.environ["AGNO_API_KEY"] = agno_api_key
        if agno_monitor is not None:
            os.environ["AGNO_MONITOR"] = agno_monitor

        super().__init__(
            name=self.cfg.name,
            role=self.cfg.role,
            model=self.llm,
            description=self.cfg.description,
            instructions=[self.cfg.system_message],
            markdown=True,
            monitoring=True,
            show_tool_calls=True,
            tools=tools,
        )

    @abstractmethod
    def run_query(self, query: str, **kwargs) -> Any:
        return super().run(message=query, **kwargs)
