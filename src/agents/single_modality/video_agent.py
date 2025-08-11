import os
from typing import Any, List, Optional

from agents.base_agent import BaseAgent
from agno.tools.serpapi import SerpApiTools
from agno.utils.log import logger
from omegaconf import DictConfig
from utils.general_utils import extract_video_urls


class VideoAgent(BaseAgent):
    def __init__(
        self,
        cfg: DictConfig,
        llm,
        tools: Optional[List[Any]] = None,
    ):
        tools = (
            [SerpApiTools(api_key=os.getenv("SERP_API_KEY"), search_youtube=True)]
            if tools is None
            else tools
        )
        super().__init__(cfg=cfg.video_agent, logger=logger, llm=llm, tools=tools)

    def run_query(self, query: str, **kwargs):
        logger.info(f"Looking up videos with query: {query}.")
        response = super().run(query)
        video_urls = extract_video_urls(text=response.content)

        if not video_urls:
            logger.warning("No video URLs found.")

        top_url = video_urls[0]
        logger.info(f"URL of video: {[top_url]}")

        return top_url
