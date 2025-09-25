import logging
import os
from typing import Dict, List

from crewai import LLM, Agent, Task
from crewai_tools import TavilySearchTool
from omegaconf import DictConfig

from tools.image_search import image_search
from tools.video_search import video_search


class Orchestrator:
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm: LLM,
        paraphrase_queries: Dict[str, str],
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.llm = llm
        self.paraphrase_queries = paraphrase_queries
        self.tasks: List[Task] = []
        self.agents: List[Agent] = []

    def _init_text_agent_task(self, query: str) -> None:
        text_agent = Agent(
            llm=self.llm,
            tools=[
                TavilySearchTool(
                    api_key=os.getenv("TAVILY_API_KEY"),
                    max_results=self.cfg.text_agent.max_results,
                    include_images=False,
                    exclude_domains=["youtube.com", "youtu.be"],
                )
            ],
            **self.cfg.text_agent.agent,
        )
        self.agents.append(text_agent)
        self.tasks.append(
            Task(
                description=query,
                agent=text_agent,
                expected_output=self.cfg.text_agent.task.expected_output,
            )
        )

    def _init_image_agent_task(self, query: str) -> None:
        image_agent = Agent(
            llm=self.llm,
            tools=[image_search],
            **self.cfg.image_agent.agent,
        )
        self.agents.append(image_agent)
        self.tasks.append(
            Task(
                description=query,
                agent=image_agent,
                expected_output=self.cfg.image_agent.task.expected_output,
            )
        )

    def _init_video_agent_task(self, query: str) -> None:
        video_agent = Agent(
            llm=self.llm,
            tools=[video_search],
            **self.cfg.video_agent.agent,
        )
        self.agents.append(video_agent)
        self.tasks.append(
            Task(
                description=query,
                agent=video_agent,
                expected_output=self.cfg.video_agent.task.expected_output,
            )
        )
        
    def run(self):
        