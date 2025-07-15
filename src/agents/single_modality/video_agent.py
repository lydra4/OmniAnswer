import logging
import os
from typing import Any, List, Optional

from agents.base_agent import BaseAgent
from agno.tools.serpapi import SerpApiTools
from omegaconf import DictConfig
from utils.general_utils import extract_video_urls


class VideoAgent(BaseAgent):
    """
    Agent for video processing tasks.
    """

    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
        tools: Optional[List[Any]] = None,
    ):
        tools = (
            [SerpApiTools(api_key=os.getenv("SERP_API_KEY"), search_youtube=True)]
            if tools is None
            else tools
        )
        super().__init__(cfg=cfg.video_agent, logger=logger, llm=llm, tools=tools)

    def run(self, query: str, **kwargs):
        """
        Run the VideoAgent to process the given query.

        Args:
            query (str): The input query to process.

        Returns:
            List[dict]: A list of dictionaries containing video titles and URLs.
        """
        self.logger.info(f"Looking up videos with query: {query}.")
        response = super().run(query)
        video_urls = extract_video_urls(text=response.content)

        if not video_urls:
            self.logger.warning("No video URLs found.")

        top_url = video_urls[0]
        self.logger.info(f"URL of video: {[top_url]}")

        return top_url
