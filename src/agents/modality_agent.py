"""Agent for selecting the best learning modalities for a query."""

import json
import logging
from typing import Any, List, Optional

from crewai import LLM
from crewai.tasks.task_output import TaskOutput
from crewai.tools import BaseTool
from omegaconf import DictConfig
from pydantic import BaseModel

from agents.base_agent.base_agent_task import BaseAgentTask


class ModalityAgent(BaseAgentTask):
    """Agent that routes queries to one or more learning modalities."""

    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm: LLM,
        output: BaseModel,
        tools: Optional[List[BaseTool]] = None,
    ) -> None:
        """Initialize the modality agent.

        Args:
            cfg: Configuration for the CrewAI agent and task.
            logger: Logger instance used for operational logging.
            llm: Language model used for reasoning about modality choices.
            output: Pydantic model describing the expected JSON output schema.
            tools: Optional list of tools available to this agent.
        """
        super().__init__(
            cfg=cfg,
            logger=logger,
            llm=llm,
            output=output,
            tools=tools,
        )

    def _parse_result(self, result: TaskOutput) -> List[str]:
        """Parse the task output and extract the selected modalities.

        Args:
            result: Task output returned by the CrewAI task.

        Returns:
            A list of selected modality identifiers such as ``\"text\"`` or
            ``\"video\"``.

        Raises:
            ValueError: If the result does not contain any JSON payload.
        """
        result_json = result.json
        if result_json is None:
            raise ValueError("No result found from modality agent.")

        parsed_result = json.loads(result_json)
        items: List[str] = parsed_result["items"]
        return items

    def run_query(self, query: str, **kwargs: Any) -> List[str]:
        """Infer the best learning modalities for a user query.

        Args:
            query: Natural language query describing what the user wants to learn.
            **kwargs: Additional keyword arguments forwarded to the task template.

        Returns:
            A list of recommended modalities for answering the query.
        """
        task = super().create_task(query=query, **kwargs)
        result = task.execute_sync()
        parsed_result = self._parse_result(result=result)
        self.logger.info(
            f"For the query:'{query}', best modes of learning: {parsed_result}."
        )
        return parsed_result
