import logging
import os
from typing import Any, List

from agents.base_agent import BaseAgent
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
from google_images_search import GoogleImagesSearch
from omegaconf import DictConfig


class ImageAgent(BaseAgent):
    def __init__(
        self, cfg: DictConfig, logger: logging.Logger, llm, tools: List[Any] = None
    ):
        if tools is None:
            tools = [
                DuckDuckGoTools(
                    stop_after_tool_call_tools=["duckduckgo_image"],
                    show_result_tools=["duckduckgo_image"],
                    fixed_max_results=cfg.image_agent.num,
                )
            ]
        super().__init__(cfg=cfg.image_agent, logger=logger, llm=llm, tools=tools)

    def _google_image_search(self, query: str):
        load_dotenv()
        gis = GoogleImagesSearch(
            developer_key=os.getenv("GOOGLE_API_KEY"),
            custom_search_cx=os.getenv("GOOGLE_CSE_ID"),
        )
        _search_params = {
            "q": query,
            "num": self.cfg.image_agent.num,
            "safe": "high",
            "imgType": "photo",
        }
        gis.search(search_params=_search_params)

    def run(self, query: str):
        """
        Run the ImageAgent to process the given query.

        Args:
            query (str): The input query to process.

        Returns:
            List[dict]: A list of dictionaries containing image titles and URLs.
        """
        self.logger.info(f"Running ImageAgent with query: {query}.")
        response = super().run(query)
        return response
