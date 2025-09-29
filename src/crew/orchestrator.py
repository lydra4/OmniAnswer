import logging
import os
from typing import Dict

from crewai import LLM, Agent, Crew, Process, Task
from crewai_tools import TavilySearchTool
from omegaconf import DictConfig

from schemas.schemas import StringOutput
from tools.image_search import image_search
from tools.video_search import video_search


class Orchestrator:
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm: LLM,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.llm = llm

    def text_agent(self) -> Agent:
        return Agent(
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

    def image_agent(self) -> Agent:
        return Agent(
            llm=self.llm,
            tools=[image_search],
            **self.cfg.image_agent.agent,
        )

    def video_agent(self) -> Agent:
        return Agent(
            llm=self.llm,
            tools=[video_search],
            **self.cfg.video_agent.agent,
        )

    def text_task(self) -> Task:
        return Task(
            description="{text_query}",
            agent=self.text_agent(),
            expected_output=self.cfg.text_agent.task.expected_output,
            output_pydantic=StringOutput,
        )

    def image_task(self) -> Task:
        return Task(
            description="Find {num_results} images for: {image_query}",
            agent=self.image_agent(),
            expected_output=self.cfg.image_agent.task.expected_output,
            output_pydantic=StringOutput,
        )

    def video_task(self) -> Task:
        return Task(
            description="Find {max_results} images for: {video_query}",
            agent=self.video_agent(),
            expected_output=self.cfg.video_agent.task.expected_output,
            output_pydantic=StringOutput,
        )

    def crew(self) -> Crew:
        return Crew(
            agents=[self.text_agent(), self.image_agent(), self.video_agent()],
            tasks=[self.text_task(), self.image_task(), self.video_task()],
            process=Process.sequential,
            verbose=True,
        )

    def run(self, paraphrase_queries: Dict[str, str]):
        research_crew = self.crew()
        results = research_crew.kickoff(
            inputs={
                "text_query": paraphrase_queries.get("text", ""),
                "image_query": paraphrase_queries.get("image", ""),
                "num_results": self.cfg.image_agent.num_results,
                "video_query": paraphrase_queries.get("video", ""),
                "max_results": self.cfg.video_agent.max_results,
            }
        )
        outout = {
            mode: result
            for mode, result in zip(
                paraphrase_queries.keys(),
                results,
            )
        }
        print(outout)
