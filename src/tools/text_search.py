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


class TextSearch(Toolkit):
    def __init__(self, cfg: DictConfig, **kwargs):
        self.cfg = cfg
        super().__init__(
            name=self.cfg.text_agent.tool_name, tools=[self._serpapi_search], **kwargs
        )

    def _extract_text_from_url(self, urls: List[str]) -> Dict[str, str]:
        texts: dict = {}
        for url in urls:
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Connection": "keep-alive",
                    "Referer": "https://www.google.com/",
                }
                response = requests.get(
                    url=url,
                    headers=headers,
                    timeout=10,
                )
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style"]):
                    tag.extract()
                text = soup.get_text(separator=" ", strip=True)
                words = "".join(
                    ch for ch in text if unicodedata.category(ch) != "Cf"
                ).split()
                trimmed = " ".join(words[: self.cfg.text_agent.max_words])
                texts[url] = trimmed
            except Exception as e:
                logger.warning(f"Text fetch failed for {url}: {e}.")
        return texts

    def _serpapi_search(self, query: str) -> Dict[str, str]:
        logger.info(f'Performing web search on "{query}".')
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
