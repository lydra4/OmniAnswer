import os
from typing import List

from crewai.tools import tool
from google_images_search import GoogleImagesSearch


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
    gis = GoogleImagesSearch(
        developer_key=os.getenv("GEMINI_API_KEY"),
        custom_search_cx=os.getenv("GOOGLE_CSE_ID"),
    )
    search_params = {
        "q": query,
        "num": num_results,
        "fileType": "jpg|gif|png",
        "safe": "active",
    }
    gis.search(search_params=search_params)
    images_urls = [image.url for image in gis.results()]
    return images_urls
