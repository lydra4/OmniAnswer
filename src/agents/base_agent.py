import logging
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from agno.agent import Agent
from omegaconf import DictConfig


class BaseAgent(Agent, ABC):
    """
    Abstract base class for agents using the Agno framework.

    Attributes:
        cfg (DictConfig): Configuration object for the agent.
        logger (logging.Logger): Logger instance used by the agent.
        llm: Language model instance used by the agent.
    """

    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
        tools: Optional[List[Any]] = None,
    ) -> None:
        """
        Initializes the BaseAgent with configuration, logger, LLM, and optional tools.

        Args:
            cfg (DictConfig): Configuration dictionary containing agent parameters.
            logger (logging.Logger): Logger for tracking agent activity.
            llm (Any): The LLM instance to be used by the agent.
            tools (List[Any], optional): List of tool functions to include in the agent. Defaults to None.
        """
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
    def run(self, query: str, **kwargs) -> Any:
        """
        Abstract method to run the agent with a given query.

        Subclasses must implement this method to define how the agent processes the input.

        Args:
            query (str): The input query string to be processed.
            **kwargs: Additional keyword arguments for extended functionality.

        Returns:
            Any: The result of processing the query.
        """
        return super().run(query, **kwargs)
