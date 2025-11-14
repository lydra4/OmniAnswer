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
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm: LLM,
        output: BaseModel,
        tools: Optional[List[Any]] = None,
    ) -> None:
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
        result_json = result.json
        if result_json is None:
            raise ValueError("No result found from text search.")

        parsed_result = json.loads(result_json)
        return parsed_result["url"]

    def run_query(self, query: str, **kwargs) -> str:
        task = super().create_task(
            query=query, max_results=self.cfg.max_results, **kwargs
        )
        result = task.execute_sync()
        parsed_result = self._parse_result(result=result)
        self.logger.info(f"For text: '{parsed_result}'.")
        return parsed_result
