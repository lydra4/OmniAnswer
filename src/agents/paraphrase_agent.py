"""Agent for generating modality‑specific paraphrases of a query."""

import json
import logging
from typing import Any, Dict, List, Optional

from crewai import LLM
from crewai.tasks.task_output import TaskOutput
from crewai.tools import BaseTool
from omegaconf import DictConfig
from pydantic import BaseModel

from agents.base_agent.base_agent_task import BaseAgentTask


class ParaphraseAgent(BaseAgentTask):
    """Agent that produces paraphrases tailored to different modalities."""

    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm: LLM,
        output: BaseModel,
        tools: Optional[List[BaseTool]] = None,
    ) -> None:
        """Initialize the paraphrase agent.

        Args:
            cfg: Configuration for the agent and task.
            logger: Logger instance used for operational logging.
            llm: Language model powering paraphrase generation.
            output: Pydantic model describing the expected JSON output schema.
            tools: Optional list of tools used during paraphrasing.
        """
        super().__init__(
            cfg=cfg,
            logger=logger,
            llm=llm,
            output=output,
            tools=tools,
        )

    def _parse_result(self, result: TaskOutput) -> Dict[str, str]:
        """Parse the task output and extract modality‑specific paraphrases.

        Args:
            result: Task output returned by the CrewAI task.

        Returns:
            A mapping from modality name (for example, ``\"text\"`` or ``\"video\"``)
            to its paraphrased query.

        Raises:
            ValueError: If the result does not contain any JSON payload.
        """
        result_json = result.json
        if result_json is None:
            raise ValueError("No result found from paraphrase task.")

        parsed_result = json.loads(result_json)
        items: Dict[str, str] = parsed_result["items"]
        return items

    def run_query(self, query: str, **kwargs: Any) -> Dict[str, str]:
        """Generate paraphrases for each requested modality.

        Args:
            query: Original user query describing what they want to learn.
            **kwargs: Additional parameters, expected to include ``modalities``.

        Returns:
            A dictionary mapping modality names to paraphrased queries.

        Raises:
            ValueError: If the ``modalities`` argument is empty or missing.
        """
        modalities = kwargs.get("modalities", [])
        if not modalities:
            raise ValueError("Modalities is empty.")

        task = super().create_task(query=query, modalities=modalities)
        result = task.execute_sync()
        parsed_result = self._parse_result(result=result)
        self.logger.info(f'Paraphrase results: "{parsed_result}"')
        return parsed_result
