"""Crew-based orchestration of text, image, and video retrieval agents."""

import logging
import os
from typing import Dict, List, cast

from crewai import LLM, Agent, Crew, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.crews.crew_output import CrewOutput
from crewai_tools import TavilySearchTool
from omegaconf import DictConfig

from schemas.schemas import ResultDictFile, ResultItem, StringOutput
from tools.image_search import ImageSearchTool
from tools.video_search import VideoSearchTool


class Orchestrator:
    """Coordinate modality-specific agents and aggregate their results."""

    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm: LLM,
    ) -> None:
        """Initialize the orchestrator and its underlying tools.

        Args:
            cfg: Configuration containing agent and task definitions.
            logger: Logger instance for reporting orchestration events.
            llm: Shared language model used by all underlying agents.
        """
        self.cfg = cfg
        self.logger = logger
        self.llm = llm

        self.text_search: TavilySearchTool | None = None
        self.image_search: ImageSearchTool | None = None
        self.video_search: VideoSearchTool | None = None

    def text_agent(self) -> BaseAgent:
        """Create the text-focused research agent."""
        if self.text_search is None:
            raise RuntimeError("text_search tool not initialized")
        return Agent(
            llm=self.llm,
            tools=[self.text_search],
            **self.cfg.text_agent.agent,
        )

    def image_agent(self) -> BaseAgent:
        """Create the image-focused research agent."""
        if self.image_search is None:
            raise RuntimeError("image_search tool not initialized")
        return Agent(
            llm=self.llm,
            tools=[self.image_search],
            **self.cfg.image_agent.agent,
        )

    def video_agent(self) -> BaseAgent:
        """Create the video-focused research agent."""
        if self.video_search is None:
            raise RuntimeError("video_search tool not initialized")
        return Agent(
            llm=self.llm,
            tools=[self.video_search],
            **self.cfg.video_agent.agent,
        )

    def text_task(self) -> Task:
        """Create the text retrieval task definition."""
        return Task(
            description=self.cfg.text_agent.task.description,
            agent=self.text_agent(),
            expected_output=self.cfg.text_agent.task.expected_output,
            output_pydantic=StringOutput,
        )

    def image_task(self) -> Task:
        """Create the image retrieval task definition."""
        return Task(
            description=self.cfg.image_agent.task.description,
            agent=self.image_agent(),
            expected_output=self.cfg.image_agent.task.expected_output,
            output_pydantic=StringOutput,
        )

    def video_task(self) -> Task:
        """Create the video retrieval task definition."""
        return Task(
            description=self.cfg.video_agent.task.description,
            agent=self.video_agent(),
            expected_output=self.cfg.video_agent.task.expected_output,
            output_pydantic=StringOutput,
        )

    def _init_tools(self, modalities: List[str]) -> None:
        if "text" in modalities:
            self.text_search = TavilySearchTool(
                api_key=os.getenv("TAVILY_API_KEY"),
                max_results=self.cfg.text_agent.max_results,
                include_images=False,
                exclude_domains=["youtube.com", "youtu.be"],
            )
        if "image" in modalities:
            self.image_search = ImageSearchTool(_cfg=self.cfg.image_agent)
        if "video" in modalities:
            self.video_search = VideoSearchTool(_cfg=self.cfg.video_agent)

    def _crew(self, modalities: List[str]) -> Crew:
        """Assemble a Crew comprising all modality-specific agents and tasks."""
        agent_map = {
            "text": self.text_agent,
            "image": self.image_agent,
            "video": self.video_agent,
        }
        task_map = {
            "text": self.text_task,
            "image": self.image_task,
            "video": self.video_task,
        }
        selected_agents = [agent_map[m]() for m in modalities]
        selected_tasks = [task_map[m]() for m in modalities]
        return Crew(agents=selected_agents, tasks=selected_tasks)

    async def run(
        self,
        query: str,
        paraphrase_queries: Dict[str, str],
    ) -> ResultDictFile:
        """Kick off the crew and aggregate results into a structured dictionary.

        Args:
            query: Original user query.
            paraphrase_queries: Mapping from modality to paraphrased query text.

        Returns:
            A :class:`ResultDictFile` object containing modality, paraphrase, and
            URL tuples.
        """
        modalities = list(paraphrase_queries.keys())
        self._init_tools(modalities=modalities)
        research_crew = self._crew(modalities=modalities)
        crew_results = cast(
            CrewOutput,
            await research_crew.kickoff_async(
                inputs={
                    "text_query": paraphrase_queries.get("text", ""),
                    "image_query": paraphrase_queries.get("image", ""),
                    "video_query": paraphrase_queries.get("video", ""),
                }
            ),
        )
        tasks_output = crew_results.tasks_output
        results_dict = {
            mode: cast(StringOutput, result.pydantic).url
            for mode, result in zip(paraphrase_queries.keys(), tasks_output)
            if result.pydantic is not None
        }
        self.logger.info(
            f"For query:'{query}', these are the modes and links for learning: '{results_dict}'."
        )

        results: List[ResultItem] = [
            {
                "modality": mode,
                "paraphrase": paraphrase_query,
                "url": cast(StringOutput, task_output.pydantic).url,
            }
            for mode, paraphrase_query, task_output in zip(
                paraphrase_queries.keys(), paraphrase_queries.values(), tasks_output
            )
            if task_output.pydantic is not None
        ]
        result_dict: ResultDictFile = {"query": query, "results": results}
        return result_dict
