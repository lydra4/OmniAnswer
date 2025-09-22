import logging
from typing import Any, List, Optional

from crewai import LLM
from crewai_tools import YoutubeVideoSearchTool
from omegaconf import DictConfig
from pydantic import BaseModel

from agents.base_agent.base_agent_task import BaseAgentTask


class VideoAgent(BaseAgentTask):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm: LLM,
        output: BaseModel,
        tools: Optional[List[Any]] = None,
    ) -> None:
        tools = [YoutubeVideoSearchTool()] if tools is None else tools
        super().__init__(
            cfg=cfg,
            logger=logger,
            llm=llm,
            output=output,
            tools=tools,
        )

    def _parse_result(self):
        pass

    def run_query(self, query: str, **kwargs):
        task = super().create_task(query=query, **kwargs)
        result = task.execute_sync()
        print(result)
