import logging
import os
from typing import Any, List, Optional

from agents.base_agent import BaseAgent
from dotenv import load_dotenv
from omegaconf import DictConfig
from pexelsapi.pexels import Pexels


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
        load_dotenv()
        tools = [self._pexels_image_search] if tools is None else tools
        super().__init__(cfg=cfg.image_agent, logger=logger, llm=llm, tools=tools)

    def _pexels_image_search(self, query: str) -> List[str]:
        load_dotenv()
        p = Pexels(api_key=os.getenv("PEXELS_API_KEY"))
        results = p.search_photos(query=query, per_page=self.cfg.max_results)["photos"]

        return [result["src"]["original"] for result in results]

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
        print(response.content)
