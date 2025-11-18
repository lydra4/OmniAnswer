"""Custom CrewAI tool for searching educational videos on YouTube."""

import os
from typing import List, Type

from crewai.tools import BaseTool
from googleapiclient.discovery import build
from omegaconf import DictConfig
from pydantic import BaseModel, Field, PrivateAttr


class VideoSearchSchema(BaseModel):
    """Schema describing the inputs to the video search tool."""

    query: str = Field(description="The search query for finding youtube videos.")


class VideoSearchTool(BaseTool):
    """CrewAI tool that wraps the YouTube Data API for video search."""

    name: str
    description: str
    serviceName: str = "youtube"
    version: str = "v3"
    developerKey: str | None = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY"),
        description="API Key for searching Youtube videos",
    )
    _cfg: DictConfig = PrivateAttr()
    _youtube: build = PrivateAttr()
    args_schema: Type[BaseModel] = VideoSearchSchema

    def __init__(self, _cfg: DictConfig) -> None:
        """Create a new video search tool instance.

        Args:
            _cfg: Configuration object containing tool metadata and limits such as
                ``max_results``.
        """
        super().__init__(name=_cfg.tool.name, description=_cfg.tool.description)
        self._youtube = build(
            serviceName=self.serviceName,
            version=self.version,
            developerKey=self.developerKey,
        )
        self._cfg = _cfg

    def _run(self, query: str) -> List[str]:
        """Execute a YouTube search for the given query.

        Args:
            query: Natural language query describing the desired videos.

        Returns:
            A list of YouTube watch URLs corresponding to the search results.
        """
        results = (
            self._youtube.search()
            .list(
                q=query,
                maxResults=self._cfg.tool.max_results,
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
