import logging
import os
from typing import Any, List

from agents.base.base_agent import BaseAgent
from dotenv import load_dotenv
from googleapiclient.discovery import build
from omegaconf import DictConfig
from utils.general_utils import extract_video_titles_and_urls


class VideoAgent(BaseAgent):
    """
    Agent for video processing tasks.
    """

    def __init__(
        self, cfg: DictConfig, logger: logging.Logger, llm, tools: List[Any] = None
    ):
        tools = [self._youtube_search] if tools is None else tools
        super().__init__(cfg=cfg.video_agent, logger=logger, llm=llm, tools=tools)

    def _youtube_search(self, query: str):
        load_dotenv()
        youtube = build(
            serviceName="youtube",
            version="v3",
            developerKey=os.getenv("GEMINI_API_KEY"),
        )
        request = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=self.cfg.num,
        )
        return request.execute()

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
        # print(response.content)
        matches = extract_video_titles_and_urls(text=response.content)
        result = [{"title": title, "url": url} for title, url in matches]
        self.logger.info(f"URL of videos: {result}.")
        return result
