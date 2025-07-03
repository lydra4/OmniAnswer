import logging
from typing import Any, List, Optional

from agents.single_modality.image_agent import ImageAgent
from agents.single_modality.text_agent import TextAgent
from agents.single_modality.video_agent import VideoAgent
from agno.team import Team
from omegaconf import DictConfig


class MultiModalTeam:
    def __init__(
        self,
        text_agent: Optional[TextAgent],
        image_agent: Optional[ImageAgent],
        video_agent: Optional[VideoAgent],
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
        tools: List[Any] = None,
    ) -> None:
        self.text_agent = text_agent
        self.image_agent = image_agent
        self.video_agent = video_agent
        self.cfg = cfg
        self.logger = logger
        self.llm = llm
        self.tools = tools if tools is not None else []
        self.agents = [
            agent
            for agent in [self.text_agent, self.image_agent, self.video_agent]
            if agent is not None
        ]

    def _define_team(self) -> None:
        return Team(
            name=self.cfg.name,
            mode=self.cfg.mode,
            members=self.agents,
            instructions=self.cfg.instructions,
            model=self.llm,
            expected_output=self.cfg.expected_output,
            share_member_interactions=self.cfg.share_member_interactions,
            markdown=True,
            monitoring=True,
        )

    def run(self, query: str) -> Any:
        multimodal_team = self._define_team()
        self.logger.info(f"Running MultiModalTeam on: {query}.")
        team_response = multimodal_team.run(query)
        for response in team_response.member_response:
            print(f"Agent ID: {response.agent_id}")
            print(f"Content: {response.content}")
