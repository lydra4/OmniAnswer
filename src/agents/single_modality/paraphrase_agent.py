import logging
from typing import Any, Dict, List, Optional

from jinja2 import Template
from omegaconf import DictConfig

from agents.base_agent import BaseAgent


class ParaphraseAgent(BaseAgent):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
        tools: Optional[List[Any]] = None,
    ) -> None:
        super().__init__(
            cfg=cfg.paraphrase_agent,
            logger=logger,
            llm=llm,
            tools=tools,
        )
        self.cfg = cfg
        self.raw_system_message = self.cfg.paraphrase_agent.system_message

    def _render_system_message(self, query: str, modality: str) -> str:
        template = Template(self.raw_system_message)
        rendered_prompt = template.render(query=query, modality=modality)

        return rendered_prompt

    def run_query(self, query: str, **kwargs) -> Dict[str, str]:
        modalities = kwargs.get("modalities", [])
        if not modalities:
            raise ValueError("Missing modalities in kwargs.")

        self.logger.info(
            f'Running ParaphraseAgent with query: "{query}" and modalities: "{modalities}"'
        )

        results: Dict[str, str] = {}

        for mode in modalities:
            system_prompt = self._render_system_message(query=query, modality=mode)

            try:
                response = super().run(
                    message=query, modality=mode, system_prompt=system_prompt, **kwargs
                )
                results[mode] = response.content.strip()

            except Exception as e:
                self.logger.error(f"Error processing modality {mode}: {e}")

        self.logger.info(f'Paraphrase results: "{results}"')
        return results
