import os
from typing import List

from crewai.tools import tool
from pexelsapi.pexels import Pexels


@tool("image_search")
async def image_search(query: str, num_results: int) -> List[str]:
    """
    Search for images using the Pexels API.

    Args:
        query (str): The search term (e.g., "cats", "sunset").
        num_results (int): Number of images to return.

    Returns:
        List[str]: A list of image URLs.
    """
    pexels = Pexels(api_key=os.getenv("PEXELS_API_KEY"))
    search_results = pexels.search_photos(query=query, per_page=num_results)
    images_urls = [result["src"]["medium"] for result in search_results["photos"]]
    return images_urls
