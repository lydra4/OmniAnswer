"""Abstract base class for agents that wrap CrewAI tasks."""

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from crewai import LLM, Agent, Task
from crewai.tasks.task_output import TaskOutput
from crewai.tools import BaseTool
from omegaconf import DictConfig
from pydantic import BaseModel


class BaseAgentTask(ABC):
    """Base class encapsulating common agent/task wiring logic.

    Subclasses supply a concrete :meth:`_parse_result` implementation and may
    override :meth:`create_task` if needed.
    """

    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm: LLM,
        output: type[BaseModel],
        tools: Optional[List[BaseTool]] = None,
    ) -> None:
        """Initialize the base agent/task wrapper.

        Args:
            cfg: Configuration containing ``agent`` and ``task`` settings.
            logger: Logger instance used for operational logging.
            llm: Language model used to power the underlying CrewAI agent.
            output: Pydantic model type describing the expected JSON output.
            tools: Optional list of tools made available to the agent.
        """
        self.cfg = cfg
        self.logger = logger
        self.llm = llm
        self.output = output
        self.tools = tools or []
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        """Instantiate the underlying CrewAI agent.

        Returns:
            A configured CrewAI :class:`Agent` instance.
        """
        agent = Agent(
            llm=self.llm,
            tools=self.tools,
            **self.cfg.agent,
        )
        self.logger.info(f"'{self.cfg.agent.role}' successfully initialized.")
        return agent

    @abstractmethod
    def _parse_result(self, result: TaskOutput) -> Any:
        """Convert a raw :class:`TaskOutput` into the desired Python object."""

    def create_task(self, query: str, **kwargs: Any) -> Task:
        """Create a CrewAI task using the configured template.

        All keys in ``cfg.task`` are treated as format strings and rendered using
        the provided ``query`` and any extra keyword arguments.

        Args:
            query: User query to inject into the task template.
            **kwargs: Additional values used to format the task template.

        Returns:
            A configured CrewAI :class:`Task` instance ready for execution.
        """
        rendered = {
            k: v.format(query=query, **kwargs) for k, v in self.cfg.task.items()
        }
        return Task(
            **rendered,
            agent=self.agent,
            output_json=self.output,
        )
