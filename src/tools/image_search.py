import os
from typing import Dict, List, Type, Union

from crewai.tools import BaseTool
from google_images_search import GoogleImagesSearch
from pydantic import BaseModel, Field


class ImageSearchSchema(BaseModel):
    query: str = Field(description="The search query for finding images.")


class ImageSearchTool(BaseTool):
    name: str = "Image Search"
    description: str = (
        "Searches Google Images for a given query and returns a list of image URLs."
    )
    args_schema: Type[BaseModel] = ImageSearchSchema

    def __init__(self, num_results: int) -> None:
        super().__init__()
        self.num_results = num_results
        self.gis = GoogleImagesSearch(
            developer_key=os.getenv("GEMINI_API_KEY"),
            custom_search_cx=os.getenv("GOOGLE_CSE_ID"),
        )
        self.search_params: Dict[str, Union[int, str]] = {
            "num": num_results,
            "fileType": "jpg|gif|png",
            "safe": "active",
        }

    def _run(self, query: str) -> List[str]:
        self.search_params["q"] = query
        self.gis.search(search_params=self.search_params)
        image_urls = [image.url for image in self.gis.results()]
        return image_urls
