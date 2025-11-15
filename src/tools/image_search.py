import os
from typing import Dict, List, Type, Union

from crewai.tools import BaseTool
from google_images_search import GoogleImagesSearch
from omegaconf import DictConfig
from pydantic import BaseModel, Field, PrivateAttr


class ImageSearchSchema(BaseModel):
    query: str = Field(..., description="The search query for finding images.")


class ImageSearchTool(BaseTool):
    name: str
    description: str
    developer_key: str | None = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY"),
        description="API Key for searching google image",
    )
    custom_search_cx: str | None = Field(
        default_factory=lambda: os.getenv("GOOGLE_CSE_ID"),
        description="Specify search engine configuration to use",
    )
    cfg: DictConfig
    _gis: GoogleImagesSearch = PrivateAttr()
    args_schema: Type[BaseModel] = ImageSearchSchema

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(name=cfg.tool.name, description=cfg.tool.description, cfg=cfg)
        self._gis: GoogleImagesSearch = GoogleImagesSearch(
            developer_key=self.developer_key,
            custom_search_cx=self.custom_search_cx,
        )
        self.cfg = cfg

    def _run(self, query: str) -> List[str]:
        search_params: Dict[str, Union[int, str]] = {
            "q": query,
            "num": self.cfg.num_results,
            "fileType": "jpg|gif|png",
            "safe": "active",
        }
        self._gis.search(search_params=search_params)
        image_urls = [image.url for image in self._gis.results()]
        return image_urls
