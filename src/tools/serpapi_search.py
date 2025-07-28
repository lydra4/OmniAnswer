import os
from typing import List

from agno.tools import Toolkit
from agno.utils.log import logger
from dotenv import load_dotenv
from omegaconf import DictConfig
from serpapi import GoogleSearch


class SerpAPISearch(Toolkit):
    def __init__(self, cfg: DictConfig, **kwargs):
        self.cfg = cfg
        super().__init__(
            name=self.cfg.text_agent.tool_name, tools=[self._serpapi_search], **kwargs
        )

    def _serpapi_search(self, query: str) -> List[str]:
        logger.info(f"Performing web search on {query}.")
        try:
            load_dotenv()
            search = GoogleSearch(
                {
                    "q": query,
                    "num": self.cfg.text_agent.fixed_max_results,
                    "api_key": os.getenv("SERP_API_KEY"),
                }
            )
            results = search.get_dict()

            return [result["link"] for result in results["organic_results"]]

        except Exception as e:
            logger.warning(f"Error for query:{query}, {e}.")
            return []
