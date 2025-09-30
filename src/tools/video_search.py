import os
from typing import List, Type

from crewai.tools import BaseTool
from googleapiclient.discovery import build
from omegaconf import DictConfig
from pydantic import BaseModel, Field, PrivateAttr


class VideoSearchSchema(BaseModel):
    query: str = Field(description="The search query for finding youtube videos.")


class VideoSearchTool(BaseTool):
    name: str = "Video Search"
    description: str = "Searches Youtube Videos for a given query and returns a list of Youtube Videos URLs."
    serviceName: str = "youtube"
    version: str = "v3"
    developerKey: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY"),
        description="API Key for searching Youtube videos",
    )
    cfg: DictConfig
    _youtube: build = PrivateAttr()
    args_schema: Type[BaseModel] = VideoSearchSchema

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg=cfg)
        self._youtube = build(
            serviceName=self.serviceName,
            version=self.version,
            developerKey=self.developerKey,
        )
        self.cfg = cfg

    def _run(self, query: str) -> List[str]:
        results = (
            self._youtube.search()
            .list(
                q=query,
                maxResults=self.cfg.max_results,
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
