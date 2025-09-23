import os
from typing import List

from crewai.tools import tool
from googleapiclient.discovery import build


@tool("video_search")
async def video_search(query: str, max_results: int) -> List[str]:
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
