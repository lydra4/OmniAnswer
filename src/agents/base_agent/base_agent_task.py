import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from crewai import LLM, Agent, Task
from crewai.tools import BaseTool
from omegaconf import DictConfig
from pydantic import BaseModel


class BaseAgentTask(ABC):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm: LLM,
        output: BaseModel,
        tools: Optional[List[BaseTool]] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.llm = llm
        self.output = output
        self.tools = tools or []
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        agent = Agent(
            llm=self.llm,
            tools=self.tools,
            **self.cfg.agent,
        )
        self.logger.info(f"'{self.cfg.agent.role}' successfully initialized.")
        return agent

    @abstractmethod
    def _parse_result(self, result) -> Any:
        pass

    def create_task(self, query: str, **kwargs: Any) -> Task:
        rendered = {
            k: str(v).format(query=query, **kwargs) for k, v in self.cfg.task.items()
        }
        return Task(
            **rendered,
            agent=self.agent,
            output_json=self.output,
        )
