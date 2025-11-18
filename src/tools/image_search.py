"""Custom CrewAI tool for searching images via Google Custom Search."""

import os
from typing import Dict, List, Type, Union

from crewai.tools import BaseTool
from google_images_search import GoogleImagesSearch
from omegaconf import DictConfig
from pydantic import BaseModel, Field, PrivateAttr


class ImageSearchSchema(BaseModel):
    """Schema describing the inputs to the image search tool."""

    query: str = Field(..., description="The search query for finding images.")


class ImageSearchTool(BaseTool):
    """CrewAI tool that uses Google Custom Search to retrieve image URLs."""

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
    _cfg: DictConfig = PrivateAttr()
    _gis: GoogleImagesSearch = PrivateAttr()
    args_schema: Type[BaseModel] = ImageSearchSchema

    def __init__(self, _cfg: DictConfig) -> None:
        """Create a new image search tool instance.

        Args:
            _cfg: Configuration object containing tool metadata and limits such as
                ``num_results``.
        """
        super().__init__(name=_cfg.tool.name, description=_cfg.tool.description)
        self._gis: GoogleImagesSearch = GoogleImagesSearch(
            developer_key=self.developer_key,
            custom_search_cx=self.custom_search_cx,
        )
        self._cfg = _cfg

    def _run(self, query: str) -> List[str]:
        """Execute an image search for the given query.

        Args:
            query: Natural language query describing the target image.

        Returns:
            A list of image URLs that match the query.
        """
        search_params: Dict[str, Union[int, str]] = {
            "q": query,
            "num": self._cfg.tool.num_results,
            "fileType": "jpg|gif|png",
            "safe": "active",
        }
        self._gis.search(search_params=search_params)
        image_urls = [image.url for image in self._gis.results()]
        return image_urls
