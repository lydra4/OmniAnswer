import logging
from typing import Any, List, Optional

from agents.base_agent import BaseAgent
from crewai_tools import YoutubeVideoSearchTool
from omegaconf import DictConfig
from utils.general_utils import extract_video_urls


class VideoAgent(BaseAgent):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
        tools: Optional[List[Any]] = None,
    ):
        tools = [YoutubeVideoSearchTool()] if tools is None else tools
        super().__init__(
            cfg=cfg.video_agent,
            logger=logger,
            llm=llm,
            tools=tools,
        )

    def run_query(self, query: str, **kwargs):
        self.logger.info(f"Looking up videos with query: {query}.")
        response = super().run(query)
        video_urls = extract_video_urls(text=response.content)

        if not video_urls:
            self.logger.warning("No video URLs found.")

        top_url = video_urls[0]
        self.logger.info(f"URL of video: {[top_url]}")

        return top_url
