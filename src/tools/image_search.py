import os
from io import BytesIO
from typing import Dict, List

import requests
from agno.tools import Toolkit
from agno.utils.log import logger
from dotenv import load_dotenv
from google_images_search import GoogleImagesSearch
from omegaconf import DictConfig
from PIL import Image
from PIL.Image import Image as PILImage


class ImageSearch(Toolkit):
    def __init__(self, cfg: DictConfig, **kwargs):
        self.cfg = cfg
        super().__init__(
            name=self.cfg.image_agent.tool_name,
            tools=[self._image_search],
            **kwargs,
        )

    def _fetch_image(self, urls: List[str]) -> Dict[str, PILImage]:
        images: dict = {}
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    images[url] = image
            except Exception as e:
                logger.warning(f"Invalid image data from '{url}': {e}")
        return images

    def _image_search(self, query: str) -> Dict[str, PILImage]:
        logger.info(f"Performing image search on '{query}'.")
        try:
            load_dotenv()
            gis = GoogleImagesSearch(
                developer_key=os.getenv("DEVELOPER_KEY"),
                custom_search_cx=os.getenv("GOOGLE_CSE_ID"),
            )
            search_params = {
                "q": query,
                "num": 5,
                "fileType": "jpg|webp|png",
            }
            gis.search(search_params=search_params)
            results = [image.url for image in gis.results()]
            valid_images = self._fetch_image(urls=results)

            return valid_images

        except Exception as e:
            logger.warning(f"Error for query: {query}, {e}.")
            return {}
