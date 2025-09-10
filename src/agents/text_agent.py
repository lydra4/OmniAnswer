import logging
import os
from typing import Any, List, Optional

from crewai_tools import TavilySearchTool
from omegaconf import DictConfig

from agents.base_agent.base_agent_task import BaseAgentTask


class TextAgent(BaseAgentTask):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
        output,
        tools: Optional[List[Any]] = None,
    ) -> None:
        tools = (
            [
                TavilySearchTool(
                    api_key=os.getenv("TAVILY_API_KEY"),
                    max_results=cfg.n_results,
                    include_images=False,
                    exclude_domains=["youtube.com", "youtu.be"],
                ),
            ]
            if tools is None
            else tools
        )

        super().__init__(
            cfg=cfg.text_agent,
            logger=logger,
            output=output,
            llm=llm,
            tools=tools,
        )

    def run_query(self, query: str, **kwargs):
        response = super().run(message=query)
        url = response.content.strip()

        if not url.startswith("http"):
            self.logger.warning(f"Invalid response: {url}.")

        self.logger.info(f"For text: {url}.")
        return url
