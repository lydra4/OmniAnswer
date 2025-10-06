import logging
import os
from typing import Dict

from crewai import LLM, Agent, Crew, Process, Task
from crewai_tools import TavilySearchTool
from omegaconf import DictConfig

from schemas.schemas import StringOutput
from tools.image_search import ImageSearchTool
from tools.video_search import VideoSearchTool


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
        self.text_search = TavilySearchTool(
            api_key=os.getenv("TAVILY_API_KEY"),
            max_results=self.cfg.text_agent.max_results,
            include_images=False,
            exclude_domains=["youtube.com", "youtu.be"],
        )
        self.image_search = ImageSearchTool(cfg=self.cfg.image_agent)
        self.video_search = VideoSearchTool(cfg=self.cfg.video_agent)

    def text_agent(self) -> Agent:
        return Agent(
            llm=self.llm,
            tools=[self.text_search],
            **self.cfg.text_agent.agent,
        )

    def image_agent(self) -> Agent:
        return Agent(
            llm=self.llm,
            tools=[self.image_search],
            **self.cfg.image_agent.agent,
        )

    def video_agent(self) -> Agent:
        return Agent(
            llm=self.llm,
            tools=[self.video_search],
            **self.cfg.video_agent.agent,
        )

    def text_task(self) -> Task:
        return Task(
            description=self.cfg.text_agent.task.description,
            agent=self.text_agent(),
            expected_output=self.cfg.text_agent.task.expected_output,
            output_pydantic=StringOutput,
        )

    def image_task(self) -> Task:
        return Task(
            description=self.cfg.image_agent.task.description,
            agent=self.image_agent(),
            expected_output=self.cfg.image_agent.task.expected_output,
            output_pydantic=StringOutput,
        )

    def video_task(self) -> Task:
        return Task(
            description=self.cfg.video_agent.task.description,
            agent=self.video_agent(),
            expected_output=self.cfg.video_agent.task.expected_output,
            output_pydantic=StringOutput,
        )

    def crew(self) -> Crew:
        return Crew(
            agents=[self.text_agent(), self.image_agent(), self.video_agent()],
            tasks=[self.text_task(), self.image_task(), self.video_task()],
            process=Process.sequential,
        )

    def run(
        self,
        query: str,
        paraphrase_queries: Dict[str, str],
    ):
        research_crew = self.crew()
        results = research_crew.kickoff(
            inputs={
                "text_query": paraphrase_queries.get("text", ""),
                "image_query": paraphrase_queries.get("image", ""),
                "video_query": paraphrase_queries.get("video", ""),
            }
        )
        tasks_output = results.tasks_output
        results_dict = {
            mode: result.pydantic.url
            for mode, result in zip(paraphrase_queries.keys(), tasks_output)
        }
        self.logger.info(
            f"For query:'{query}', these are the modes and links for learning: '{results_dict}'."
        )
