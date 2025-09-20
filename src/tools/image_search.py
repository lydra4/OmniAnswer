from typing import Dict, List

from crewai.tools import BaseTool
from pexelsapi.pexels import Pexels


class ImageSearch(BaseTool):
    name: str = "Image Search"
    description: str = "Image Search Tool"

    def __init__(self, api_key: str, num_results: int):
        super().__init__()
        self.client: Pexels = Pexels(api_key=api_key)
        self.num_results = num_results

    def _run(self, query: str, run_manager=None) -> List[str]:
        search_results = self.client.search_photos(query=query, per_page=self.num_results)
        images_urls = [result["src"]["medium"] for result in search_results["photos"]]
        return images_urls

    def async _arun(self, query:str, run_manager=None) - > List[str]:
        return self._run(query=query, run_manager=run_manager)