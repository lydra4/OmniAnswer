import os
from typing import List, Type

from crewai.tools import BaseTool
from googleapiclient.discovery import build
from pydantic import BaseModel, Field


class VideoSearchSchema(BaseModel):
    query: str = Field(description="The search query for finding youtube videos.")


class VideoSearchTool(BaseTool):
    name: str = "Video Search"
    description: str = "Searches Youtube Videos for a given query and returns a list of Youtube Videos URLs."
    args_schema: Type(BaseModel) = VideoSearchSchema

    def __init__(self, max_results: int) -> None:
        super().__init__()
        self.max_results = max_results
        self.youtube = build(
            serviceName="youtube",
            version="v3",
            developerKey=os.getenv("GEMINI_API_KEY"),
        )

    def _run(self, query: str) -> List[str]:
        results = (
            self.youtube.search()
            .list(
                q=query,
                maxResults=self.max_results,
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
