import os
from typing import List

from agno.tools import Toolkit
from agno.utils.log import logger
from dotenv import load_dotenv
from omegaconf import DictConfig
from pexelsapi.pexels import Pexels


class PexelSearch(Toolkit):
    def __init__(self, cfg: DictConfig, **kwargs):
        self.cfg = cfg
        super().__init__(
            name=self.cfg.image_agent.tool_name,
            tools=[self._pexels_image_search],
            **kwargs,
        )

    def _pexels_image_search(self, query: str) -> List[str]:
        logger.info(f"Performing image search on '{query}'.")
        try:
            load_dotenv()
            p = Pexels(api_key=os.getenv("PEXELS_API_KEY"))
            results = p.search_photos(
                query=query,
                per_page=self.cfg.max_results,
                locale="en-US",
            )["photos"]

            return [result["url"] for result in results]

        except Exception as e:
            logger.warning(f"Error for query: {query}, {e}.")
            return []
