import os
import unicodedata
from typing import Dict, List

import requests
from agno.tools import Toolkit
from agno.utils.log import logger
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from omegaconf import DictConfig
from serpapi import GoogleSearch


class SerpAPISearch(Toolkit):
    def __init__(self, cfg: DictConfig, **kwargs):
        self.cfg = cfg
        super().__init__(
            name=self.cfg.text_agent.tool_name, tools=[self._serpapi_search], **kwargs
        )

    def _extract_text_from_url(self, urls: List[str]) -> Dict[str, str]:
        texts: dict = {}
        for url in urls:
            try:
                response = requests.get(url=url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style"]):
                    tag.extract()
                text = soup.get_text(separator=" ", strip=True)
                cleaned = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
                texts[url] = cleaned
            except Exception as e:
                logger.warning(f"Text fetch failed for {url}: {e}.")
        return texts

    def _serpapi_search(self, query: str) -> Dict[str, str]:
        logger.info(f"Performing web search on '{query}'.")
        try:
            load_dotenv()
            search = GoogleSearch(
                {
                    "q": query,
                    "num": self.cfg.text_agent.fixed_max_results,
                    "api_key": os.getenv("SERP_API_KEY"),
                }
            )
            results_dict = search.get_dict()
            urls = [result["link"] for result in results_dict["organic_results"]]
            urls_dict = self._extract_text_from_url(urls=urls)

            return urls_dict

        except Exception as e:
            logger.warning(f"Error for query:{query}, {e}.")
            return {}
