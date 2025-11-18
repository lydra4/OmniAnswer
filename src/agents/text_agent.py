"""Agents for retrieving text resources given a learning query."""

import json
import logging
import os
from typing import Any, List, Optional

from crewai import LLM
from crewai.tasks.task_output import TaskOutput
from crewai_tools import TavilySearchTool
from omegaconf import DictConfig
from pydantic import BaseModel

from agents.base_agent.base_agent_task import BaseAgentTask


class TextAgent(BaseAgentTask):
    """Agent that searches the web for text resources.

    The agent uses a Tavily-based search tool and parses the result into a single
    representative URL.
    """
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm: LLM,
        output: BaseModel,
        tools: Optional[List[Any]] = None,
    ) -> None:
        """Initialize the text agent and its search tools.

        Args:
            cfg: Configuration for the agent and task behaviour.
            logger: Logger instance used for operational logging.
            llm: Language model used for task orchestration.
            output: Pydantic model describing the expected JSON output schema.
            tools: Optional list of tools to use instead of the default Tavily
                search tool.
        """
        tools = (
            [
                TavilySearchTool(
                    api_key=os.getenv("TAVILY_API_KEY"),
                    max_results=cfg.max_results,
                    include_images=False,
                    exclude_domains=["youtube.com", "youtu.be"],
                ),
            ]
            if tools is None
            else tools
        )

        super().__init__(
            cfg=cfg,
            logger=logger,
            llm=llm,
            output=output,
            tools=tools,
        )
        self.cfg = cfg

    def _parse_result(self, result: TaskOutput) -> str:
        """Parse the task output and extract the target URL.

        Args:
            result: Task output returned by the CrewAI task.

        Returns:
            The URL extracted from the JSON payload.

        Raises:
            ValueError: If the result does not contain any JSON payload.
        """
        result_json = result.json
        if result_json is None:
            raise ValueError("No result found from text search.")

        parsed_result = json.loads(result_json)
        url: str = parsed_result["url"]
        return url

    def run_query(self, query: str, **kwargs: Any) -> str:
        """Execute a text search for the given query.

        Args:
            query: Natural language query describing the concept to learn.
            **kwargs: Additional parameters forwarded to the underlying task.

        Returns:
            URL of the best matching text resource.
        """
        task = super().create_task(
            query=query, max_results=self.cfg.max_results, **kwargs
        )
        result = task.execute_sync()
        parsed_result = self._parse_result(result=result)
        self.logger.info(f"For text: '{parsed_result}'.")
        return parsed_result
