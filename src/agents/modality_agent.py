import json
import logging
from typing import Any, List, Optional

from crewai import LLM, TaskOutput
from crewai.tasks.task_output import TaskOutput
from crewai.tools import BaseTool
from omegaconf import DictConfig
from pydantic import BaseModel

from agents.base_agent.base_agent_task import BaseAgentTask


class ModalityAgent(BaseAgentTask):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm: LLM,
        output: BaseModel,
        tools: Optional[List[BaseTool]] = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            logger=logger,
            llm=llm,
            output=output,
            tools=tools,
        )

    def _parse_result(self, result: TaskOutput) -> List[str]:
        result_json = result.json
        if result_json is None:
            raise ValueError("No result found from modality agent.")

        parsed_result = json.loads(result_json)
        items: List[str] = parsed_result["items"]
        return items

    def run_query(self, query: str, **kwargs: Any) -> List[str]:
        task = super().create_task(query=query, **kwargs)
        result = task.execute_sync()
        parsed_result = self._parse_result(result=result)
        self.logger.info(
            f"For the query:'{query}', best modes of learning: {parsed_result}."
        )
        return parsed_result
