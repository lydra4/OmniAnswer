import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from crewai import Agent
from crewai.tools import BaseTool
from omegaconf import DictConfig


class BaseAgent(Agent, ABC):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
        tools: Optional[List[BaseTool]] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.llm = llm

        super().__init__(
            role=self.cfg.role,
            goal=self.cfg.goal,
            backstory=self.cfg.backstory,
            llm=self.llm,
            tools=[tools],
            verbose=True,
        )

    @abstractmethod
    def run_query(self, query: str, **kwargs) -> Any:
        return super().kickoff(message=query, **kwargs)
