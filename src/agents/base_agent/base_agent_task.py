import logging
from typing import List, Optional

from crewai import Agent, Task
from crewai.project import agent, task
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
        self.tools = tools

    @agent
    def _init_agent(self) -> Agent:
        return Agent(**self.cfg.agent)

    @task
    def _init_task(self) -> Task:
        return Task(**self.cfg.task)
