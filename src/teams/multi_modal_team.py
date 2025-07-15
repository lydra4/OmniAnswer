import logging
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from agents.single_modality.image_agent import ImageAgent
from agents.single_modality.text_agent import TextAgent
from agents.single_modality.video_agent import VideoAgent
from agno.team import Team
from omegaconf import DictConfig
from pydantic import BaseModel


class ModalityLinks(BaseModel):
    text: Optional[str]
    image: Optional[str]
    video: Optional[str]


class MultiModalTeam:
    def __init__(
        self,
        paraphrased_outputs: Dict[str, str],
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
        tools: Optional[List[Any]] = None,
    ) -> None:
        self.paraphrased_outputs = paraphrased_outputs
        self.cfg = cfg
        self.logger = logger
        self.llm = llm
        self.tools = tools if tools is not None else []
        self.modality_agent_map: Dict[str, BaseAgent] = {
            "text": TextAgent(cfg=self.cfg, logger=logger, llm=llm),
            "image": ImageAgent(cfg=self.cfg, logger=logger, llm=llm),
            "video": VideoAgent(cfg=self.cfg, logger=logger, llm=llm),
        }
        self.agents: List[BaseAgent] = [
            self.modality_agent_map[modality]
            for modality in self.paraphrased_outputs
            if modality in self.modality_agent_map
        ]

    def _define_team(self) -> Team:
        return Team(
            name=self.cfg.omni_team.name,
            mode=self.cfg.omni_team.mode,
            members=self.agents,
            model=self.llm,
            share_member_interactions=self.cfg.omni_team.share_member_interactions,
            markdown=True,
            monitoring=True,
            enable_session_summaries=True,
            stream_intermediate_steps=True,
            response_model=ModalityLinks,
        )

    def run(self, query: str):
        multimodal_team = self._define_team()
        self.logger.info(f"Running MultiModalTeam on: {query}.")
        response = multimodal_team.run(query, stream=False)

        output = {
            key: value
            for key, value in {
                "text": response.content.text,
                "image": response.content.image,
                "video": response.content.video,
            }.items()
            if value is not None
        }
        self.logger.info(f"MultiModalTeam output: {output}")
        return output
