import os
from typing import List

from crewai.tools import tool
from googleapiclient.discovery import build


@tool("video_search")
async def video_search(query: str, max_results: int) -> List[str]:
    """
    Search for YouTube videos matching a given query.

    This function uses the YouTube Data API v3 to perform a video search
    based on the provided query string. It returns a list of direct YouTube
    video URLs limited by the specified number of results.

    Args:
        query (str): The search query term (e.g., "agentic workflow architecture").
        max_results (int): The maximum number of video results to return.

    Returns:
        List[str]: A list of YouTube video URLs that match the query.

    Notes:
        - Requires the environment variable `GEMINI_API_KEY` to be set
          with a valid YouTube Data API v3 key.
        - If the API request fails or no results are found, the returned list
          will be empty.
    """
    youtube = build(
        serviceName="youtube",
        version="v3",
        developerKey=os.getenv("GEMINI_API_KEY"),
    )
    results = (
        youtube.search()
        .list(
            q=query,
            maxResults=max_results,
            part="snippet",
        )
        .execute()
    )
    results_list = results["items"]
    video_list = [
        f"https://www.youtube.com/watch?v={result['id']['videoId']}"
        for result in results_list
    ]
    return video_list
