import logging
from typing import List, Optional

from crewai import Agent, Task
from crewai.tools import BaseTool
from omegaconf import DictConfig


class BaseAgentTask:
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
        self.tools = tools or []
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        return Agent(
            llm=self.llm,
            tools=self.tools,
            **self.cfg.agent,
        )

    def create_task(self, query: str, **kwargs) -> Task:
        rendered = {
            k: str(v).format(query=query, **kwargs) for k, v in self.cfg.task.items()
        }
        return Task(
            **rendered,
            agent=self.agent,
        )
