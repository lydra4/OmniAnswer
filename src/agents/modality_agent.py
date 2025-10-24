import json
import logging
from typing import List, Optional

from agents.base_agent.base_agent_task import BaseAgentTask
from crewai import LLM
from crewai.tools import BaseTool
from omegaconf import DictConfig
from pydantic import BaseModel


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

    def _parse_result(self, result: str) -> List[str]:
        result_json = result.json
        parsed_result = json.loads(result_json)
        return parsed_result["items"]

    def run_query(self, query: str, **kwargs) -> List[str]:
        task = super().create_task(query=query, **kwargs)
        result = task.execute_sync()
        parsed_result = self._parse_result(result=result)
        self.logger.info(
            f"For the query:'{query}', best modes of learning: {parsed_result}."
        )
        return parsed_result
