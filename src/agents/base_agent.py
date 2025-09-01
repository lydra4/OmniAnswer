import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from crewai import Agent
from crewai.tools import BaseTool
from omegaconf import DictConfig
from pydantic import PrivateAttr


class BaseAgent(Agent, ABC):
    model_config = {"extra": "allow"}

    _cfg: DictConfig = PrivateAttr()
    _logger: logging.Logger = PrivateAttr()
    _llm: Any = PrivateAttr()

    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
        tools: Optional[List[BaseTool]] = None,
    ) -> None:
        super().__init__(
            role=cfg.role,
            goal=cfg.goal,
            backstory=cfg.backstory,
            llm=llm,
            tools=tools or [],
            verbose=False,
        )
        object.__setattr__(self, "_cfg", cfg)
        object.__setattr__(self, "_logger", logger)
        object.__setattr__(self, "_llm", llm)

    @abstractmethod
    def run_query(self, query: str, **kwargs) -> Any:
        return super().kickoff(messages=query, **kwargs)
