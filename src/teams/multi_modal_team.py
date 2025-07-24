import json
import logging
import os
from typing import Dict, List, Optional, Union

from agents.base_agent import BaseAgent
from agents.single_modality.image_agent import ImageAgent
from agents.single_modality.text_agent import TextAgent
from agents.single_modality.video_agent import VideoAgent
from agno.agent import Agent
from agno.team import Team
from omegaconf import DictConfig
from pydantic import BaseModel


class ModalityLinks(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None
    video: Optional[str] = None


class MultiModalTeam:
    def __init__(
        self,
        paraphrased_outputs: Dict[str, str],
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
    ) -> None:
        self.paraphrased_outputs = paraphrased_outputs
        self.cfg = cfg
        self.logger = logger
        self.llm = llm
        self.modality_agent_map: Dict[str, BaseAgent] = {
            "text": TextAgent(cfg=self.cfg, logger=logger, llm=llm),
            "image": ImageAgent(cfg=self.cfg, logger=logger, llm=llm),
            "video": VideoAgent(cfg=self.cfg, logger=logger, llm=llm),
        }
        self.agents: List[Union[Agent, Team]] = [
            self.modality_agent_map[modality]
            for modality in self.paraphrased_outputs
            if modality in self.modality_agent_map
        ]
        self.team = self._define_team()
        self.file_path: str = self.cfg.omni_team.file_path

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

    def _save_output(self, file_path: str, new_output: dict):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([new_output], f, indent=4)
        else:
            with open(file_path, encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
                except json.JSONDecodeError:
                    data = []

            data.append(new_output)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

    def run(self, query: str):
        self.logger.info(f"Running MultiModalTeam on: {query}.")
        response = self.team.run(query, stream=False)

        output = {
            "original_query": query,
            "paraphrased_queries": self.paraphrased_outputs,
            "results": {
                key: value
                for key, value in {
                    "text": response.content.text,
                    "image": response.content.image,
                    "video": response.content.video,
                }.items()
                if value is not None
            },
        }

        self._save_output(file_path=self.file_path, new_output=output)

        self.logger.info(f"MultiModalTeam output: {output}")
        return output
