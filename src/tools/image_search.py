import os
from typing import List

from crewai.tools import BaseTool
from pexelsapi.pexels import Pexels


class ImageSearch(BaseTool):
    name: str = "Image Search"
    description: str = "Image Search Tool"

    def __init__(self, num_results: int):
        super().__init__()
        self.client: Pexels = Pexels(api_key=os.getenv("PEXELS_API_KEY"))
        self.num_results = num_results

    async def _run(self, query: str) -> List[str]:
        search_results = self.client.search_photos(
            query=query, per_page=self.num_results
        )
        images_urls = [result["src"]["medium"] for result in search_results["photos"]]
        return images_urls
