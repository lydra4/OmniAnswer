import logging
from typing import List, Optional

from crewai.tools import BaseTool
from omegaconf import DictConfig

from agents.base_agent.base_agent_task import BaseAgentTask
from utils.general_utils import parse_json_list


class ModalityAgent(BaseAgentTask):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
        output,
        tools: Optional[List[BaseTool]] = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            logger=logger,
            llm=llm,
            output=output,
            tools=tools,
        )

    def run_query(self, query: str, **kwargs) -> List[str]:
        self.logger.info(f"Running on query: '{query}'.")
        task = super().create_task(query=query, **kwargs)
        result = task.execute_sync()
        parsed_result = parse_json_list(output=result)
        self.logger.info(
            f"For the query:'{query}', best modes of learning: {parsed_result}."
        )
        return parsed_result
