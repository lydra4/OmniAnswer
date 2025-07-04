import json
import logging
import re
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from agents.single_modality.image_agent import ImageAgent
from agents.single_modality.text_agent import TextAgent
from agents.single_modality.video_agent import VideoAgent
from agno.team import Team
from omegaconf import DictConfig


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

    def _extract_urls(self, text: str) -> List[str]:
        return re.findall(r"(https?://\S+)", text)

    def _define_team(self) -> Team:
        return Team(
            name=self.cfg.multimodal_team.name,
            mode=self.cfg.multimodal_team.mode,
            members=self.agents,
            instructions=self.cfg.multimodal_team.instructions,
            model=self.llm,
            expected_output=self.cfg.multimodal_team.expected_output,
            share_member_interactions=self.cfg.multimodal_team.share_member_interactions,
            markdown=True,
            monitoring=True,
            enable_session_summaries=True,
            stream_intermediate_steps=True,
        )

    def run(self, query: str):
        multimodal_team = self._define_team()
        self.logger.info(f"Running MultiModalTeam on: {query}.")
        response = multimodal_team.run(query)
        content = response.content or ""
        outputs: Dict[str, List[str]] = {}

        json_match = re.search(r"```json\n(.*?)```", content, re.DOTALL)
        if json_match:
            try:
                plan = json.loads(json_match.group(1))
                for entry in plan:
                    agent_id = entry["agent_id"]  # e.g., text-search-agent
                    modality = agent_id.split("-")[0]  # "text", "image", etc.
                    task = entry["task"]
                    agent = self.modality_agent_map.get(modality)
                    if agent:
                        try:
                            result = agent.run(task)
                            # Each agent returns a list of dicts with 'url' or a list of URLs
                            if isinstance(result, list):
                                # Try to extract URLs from dicts or use as-is if already URLs
                                urls = []
                                for item in result:
                                    if isinstance(item, dict) and "url" in item:
                                        urls.append(item["url"])
                                    elif isinstance(item, str):
                                        urls.append(item)
                                outputs.setdefault(modality, []).extend(urls)
                            else:
                                outputs.setdefault(modality, []).append(result)
                        except Exception as e:
                            self.logger.error(f"Agent {modality} failed: {e}")
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse agent JSON: {e}")

        print(outputs)
        return outputs
