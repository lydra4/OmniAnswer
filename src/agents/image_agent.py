import logging
from typing import Any, List, Optional

from omegaconf import DictConfig

from agents.base_agent.base_agent_task import BaseAgentTask
from tools.image_search import image_search


class ImageAgent(BaseAgentTask):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
        tools: Optional[List[Any]] = None,
    ):
        tools = [image_search] if tools is None else tools
        super().__init__(
            cfg=cfg.image_agent,
            logger=logger,
            llm=llm,
            tools=tools,
        )

    def _parse_result(self):
        pass

    def run_query(self, query: str, **kwargs):
        response = super().run(message=query)
        url = response.content.strip()

        if not url.startswith("http"):
            self.logger.warning(f"Invalid response: {url}.")

        self.logger.info(f"For image: {url}.")
        return url
