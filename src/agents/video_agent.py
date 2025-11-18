"""Agent for retrieving educational videos relevant to a query."""

import json
import logging
from typing import Any, List, Optional

from crewai import LLM
from crewai.tasks.task_output import TaskOutput
from omegaconf import DictConfig
from pydantic import BaseModel

from agents.base_agent.base_agent_task import BaseAgentTask
from tools.video_search import video_search


class VideoAgent(BaseAgentTask):
    """Agent that uses a video search tool to retrieve video URLs."""

    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm: LLM,
        output: BaseModel,
        tools: Optional[List[Any]] = None,
    ) -> None:
        """Initialize the video agent.

        Args:
            cfg: Configuration for the agent and task.
            logger: Logger instance used for operational logging.
            llm: Language model powering reasoning for the task.
            output: Pydantic model describing the expected JSON output schema.
            tools: Optional list of tools to use instead of the default
                `video_search` tool.
        """
        tools = [video_search] if tools is None else tools
        super().__init__(
            cfg=cfg,
            logger=logger,
            llm=llm,
            output=output,
            tools=tools,
        )
        self.cfg = cfg

    def _parse_result(self, result: TaskOutput) -> str:
        """Parse the task output and extract the video URL.

        Args:
            result: Task output returned by the CrewAI task.

        Returns:
            The URL to a recommended video resource.

        Raises:
            ValueError: If the result payload is missing.
        """
        result_json = result.json
        if result_json is None:
            raise ValueError("No result found from video search.")

        parsed_result = json.loads(result_json)
        url: str = parsed_result["url"]
        return url

    def run_query(self, query: str, **kwargs: Any) -> str:
        """Execute a video search for the given query.

        Args:
            query: Natural language query describing the concept to learn.
            **kwargs: Additional parameters forwarded to the underlying task.

        Returns:
            URL of the best matching video resource.
        """
        task = super().create_task(
            query=query, num_results=self.cfg.max_results, **kwargs
        )
        result = task.execute_sync()
        parsed_result = self._parse_result(result=result)
        self.logger.info(f"For video: '{parsed_result}'.")
        return parsed_result
