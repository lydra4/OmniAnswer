import logging
from typing import Any, List

from agents.base_agent import BaseAgent
from agno.tools.youtube import YouTubeTools
from omegaconf import DictConfig


class VideoAgent(BaseAgent):
    """
    Agent for video processing tasks.
    """

    def __init__(
        self, cfg: DictConfig, logger: logging.Logger, tools: List[Any] = None
    ):
        tools = [YouTubeTools()] if tools is None else tools
        super().__init__(cfg=cfg.video_agent, logger=logger, tools=tools)

    def run(self, query: str):
        """
        Run the VideoAgent to process the given query.

        Args:
            query (str): The input query to process.

        Returns:
            List[dict]: A list of dictionaries containing video titles and URLs.
        """
        self.logger.info(f"Looking up videos with query: {query}.")
        response = super().run(query)
        result = response.content
        self.logger.info(f"URL of videos: {result}.")
        return result
