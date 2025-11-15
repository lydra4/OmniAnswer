import json
import logging
from typing import Any, List, Optional

from crewai import LLM
from crewai.tasks.task_output import TaskOutput
from omegaconf import DictConfig
from pydantic import BaseModel

from agents.base_agent.base_agent_task import BaseAgentTask
from tools.image_search import image_search


class ImageAgent(BaseAgentTask):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm: LLM,
        output: BaseModel,
        tools: Optional[List[Any]] = None,
    ):
        tools = [image_search] if tools is None else tools
        super().__init__(
            cfg=cfg,
            logger=logger,
            llm=llm,
            output=output,
            tools=tools,
        )
        self.cfg = cfg

    def _parse_result(self, result: TaskOutput) -> str:
        result_json = result.json
        if result_json is None:
            raise ValueError("No result found from image search.")

        parsed_result = json.loads(result_json)
        url: str = parsed_result["url"]
        return url

    def run_query(self, query: str, **kwargs: Any) -> str:
        task = super().create_task(
            query=query, num_results=self.cfg.tool.num_results, **kwargs
        )
        result = task.execute_sync()
        parsed_result = self._parse_result(result=result)
        self.logger.info(f"For image: '{parsed_result}'.")
        return parsed_result
