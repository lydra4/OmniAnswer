import json
import logging
from typing import Dict, List, Optional

from crewai import LLM
from crewai.tools import BaseTool
from omegaconf import DictConfig
from pydantic import BaseModel

from agents.base_agent.base_agent_task import BaseAgentTask


class ParaphraseAgent(BaseAgentTask):
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

    def _parse_result(self, result: str) -> Dict[str, str]:
        result_json = result.json
        parsed_result = json.loads(result_json)
        return parsed_result["items"]

    def run_query(self, query: str, **kwargs) -> Dict[str, str]:
        modalities = kwargs.get("modalities", [])
        if not modalities:
            raise ValueError("Modalities is empty.")

        task = super().create_task(query=query, modalities=modalities)
        result = task.execute_sync()
        parsed_result = self._parse_result(result=result)
        self.logger.info(f'Paraphrase results: "{parsed_result}"')
        return parsed_result
