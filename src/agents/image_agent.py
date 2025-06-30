import logging
import os
from typing import Any, List

from agents.base_agent import BaseAgent
from dotenv import load_dotenv
from google_images_search import GoogleImagesSearch
from omegaconf import DictConfig
from utils.general_utils import extract_image_urls


class ImageAgent(BaseAgent):
    """
    Agent responsible for retrieving relevant images based on a query using Google Custom Search API.
    """

    def __init__(
        self, cfg: DictConfig, logger: logging.Logger, llm, tools: List[Any] = None
    ):
        """
        Initializes the ImageAgent with configuration, logger, LLM, and optional tools.

        Args:
            cfg (DictConfig): Configuration specific to the image agent.
            logger (logging.Logger): Logger instance for tracking execution.
            llm (Any): The language model used to interpret or expand image search queries.
            tools (List[Any], optional): List of tools to enable (defaults to internal image search method).
        """
        tools = [self._google_image_search] if tools is None else tools
        super().__init__(cfg=cfg.image_agent, logger=logger, llm=llm, tools=tools)

    def _google_image_search(self, query: str):
        """
        Performs a Google Custom Search for images related to the input query.

        Args:
            query (str): Search term to retrieve relevant images.

        Returns:
            List[str]: A list of image URLs from the search results.
        """
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
        Processes a user query to retrieve and extract image URLs.

        Executes the configured image search tool, parses the markdown-formatted output,
        and extracts image links.

        Args:
            query (str): Text-based image search query.

        Returns:
            List[str]: List of image URLs extracted from the response content.
        """
        self.logger.info(f"Looking up images on query: {query}.")
        response = super().run(query)
        result = extract_image_urls(text=response.content)
        self.logger.info(f"URL of images: {result}.")
        return result
