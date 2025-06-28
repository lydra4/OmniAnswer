import logging
import os
import re
from typing import Any, List

from agents.base_agent import BaseAgent
from dotenv import load_dotenv
from google_images_search import GoogleImagesSearch
from omegaconf import DictConfig


class ImageAgent(BaseAgent):
    def __init__(
        self, cfg: DictConfig, logger: logging.Logger, llm, tools: List[Any] = None
    ):
        if tools is None:
            tools = [self._google_image_search]
        super().__init__(cfg=cfg.image_agent, logger=logger, llm=llm, tools=tools)

    def _google_image_search(self, query: str):
        load_dotenv()
        gis = GoogleImagesSearch(
            developer_key=os.getenv("GEMINI_API_KEY"),
            custom_search_cx=os.getenv("GOOGLE_CSE_ID"),
        )
        _search_params = {
            "q": query,
            "num": self.cfg.num,
            "safe": "active",
            "imgType": "photo",
        }
        gis.search(search_params=_search_params)
        return [image.url for image in gis.results()]

    def run(self, query: str):
        """
        Run the ImageAgent to process the given query.

        Args:
            query (str): The input query to process.

        Returns:
            List[dict]: A list of dictionaries containing image titles and URLs.
        """
        self.logger.info(f"Looking up images on query: {query}.")
        response = super().run(query)
        result = re.findall(r"\[.*?\]\((https?://.*?)\)", response.content)
        self.logger.info(f"URL of images: {result}.")
        return result
