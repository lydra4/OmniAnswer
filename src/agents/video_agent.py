import json
import logging
from typing import Any, List, Optional

from crewai import LLM
from omegaconf import DictConfig
from pydantic import BaseModel

from agents.base_agent.base_agent_task import BaseAgentTask
from tools.video_search import video_search


class VideoAgent(BaseAgentTask):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm: LLM,
        output: BaseModel,
        tools: Optional[List[Any]] = None,
    ) -> None:
        tools = [video_search] if tools is None else tools
        super().__init__(
            cfg=cfg,
            logger=logger,
            llm=llm,
            output=output,
            tools=tools,
        )
        self.cfg = cfg

    def _parse_result(self, result: str) -> str:
        result_json = result.json
        parsed_result = json.loads(result_json)
        return parsed_result["url"]

    def run_query(self, query: str, **kwargs):
        task = super().create_task(
            query=query, num_results=self.cfg.max_results, **kwargs
        )
        result = task.execute_sync()
        parsed_result = self._parse_result(result=result)
        self.logger.info(f"For video: '{parsed_result}'.")
        return parsed_result
