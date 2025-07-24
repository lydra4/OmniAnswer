import logging
import os
from typing import Any, List, Optional

import requests
from agents.base_agent import BaseAgent
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from google_images_search import GoogleImagesSearch
from omegaconf import DictConfig
from utils.general_utils import extract_image_urls


class ImageAgent(BaseAgent):
    """
    Agent responsible for retrieving relevant images based on a query using Google Custom Search API.
    """

    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
        tools: Optional[List[Any]] = None,
    ):
        """
        Initializes the ImageAgent with configuration, logger, LLM, and optional tools.

        Args:
            cfg (DictConfig): Configuration specific to the image agent.
            logger (logging.Logger): Logger instance for tracking execution.
            llm (Any): The language model used to interpret or expand image search queries.
            tools (List[Any], optional): List of tools to enable (defaults to internal image search method).
        """
        tools = [self._image_search] if tools is None else tools
        super().__init__(cfg=cfg.image_agent, logger=logger, llm=llm, tools=tools)

    def _image_search(self, query: str) -> List[str]:
        """
        Performs a Google Custom Search for images related to the input query.

        Args:
            query (str): Search term to retrieve relevant images.

        Returns:
            List[str]: A list of image URLs from the search results.
        """
        load_dotenv()
        try:
            gis = GoogleImagesSearch(
                developer_key=os.getenv("GEMINI_API_KEY"),
                custom_search_cx=os.getenv("GOOGLE_CSE_ID"),
            )
            _search_params = {
                "q": query,
                "num": self.cfg.num,
                "safe": "active",
                "imgType": "photo",
                "fileType": "jpg|png|gif|webp",
            }
            gis.search(search_params=_search_params)
            for image in gis.results():
                url = image.url
                if url and url.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".gif", ".webp")
                ):
                    try:
                        response = requests.head(url, timeout=5)
                        if response.status_code == 200:
                            return [url]
                    except requests.RequestException:
                        continue
        except Exception:
            pass

        try:
            with DDGS() as ddgs:
                for image in ddgs.images(
                    keywords=query, max_results=self.cfg.num, safesearch="On"
                ):
                    url = image.get("image")
                    if url and url.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".gif", ".webp")
                    ):
                        try:
                            response = requests.head(url=url, timeout=5)
                            if response.status_code == 200:
                                return [url]
                        except requests.RequestException:
                            continue

        except Exception:
            pass

        return []

    def run(self, query: str, **kwargs):
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
        url_result = extract_image_urls(text=response.content)
        url = " ".join(url_result)
        self.logger.info(f"URL of image: {[url]}.")

        return url
