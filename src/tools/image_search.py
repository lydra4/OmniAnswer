import os
from io import BytesIO
from typing import Dict, Optional

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
            valid_images: Dict[str, PILImage] = {}
            for url in results:
                img = self._fetch_image(url)
                if img:
                    valid_images[url] = img

            return valid_images

        except Exception as e:
            logger.warning(f"Error for query: {query}, {e}.")
            return {}

    def _fetch_image(self, url: str) -> Optional[PILImage]:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch image from '{url}': {e}")
        except Exception as e:
            logger.warning(f"Invalid image data from '{url}': {e}")
        return None
