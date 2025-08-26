import logging
import os
from typing import Any, List

from crewai.tools import BaseTool
from dotenv import load_dotenv
from google_images_search import GoogleImagesSearch
from omegaconf import DictConfig


class ImageSearch(BaseTool):
    name: str = "Image Search Tool"
    description: str = (
        "Searches Google Images and returns image URLs and fetched images."
    )

    def __init__(self, cfg: DictConfig, logger: logging.Logger, **kwargs):
        self.cfg = cfg
        self.logger = logger
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[str]:
        self.logger.info(f"Performing image search on '{query}'.")
        try:
            load_dotenv()
            gis = GoogleImagesSearch(
                developer_key=os.getenv("GEMINI_API_KEY"),
                custom_search_cx=os.getenv("GOOGLE_CSE_ID"),
            )
            search_params = {
                "q": query,
                "num": self.cfg.image_agent.max_results,
                "fileType": "jpg|webp|png",
            }
            gis.search(search_params=search_params)
            results = [image.url for image in gis.results()]

            return results

        except Exception as e:
            self.logger.warning(f"Error for query: {query}, {e}.")
            return {}
