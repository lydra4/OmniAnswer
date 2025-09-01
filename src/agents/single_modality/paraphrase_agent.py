import logging
from typing import Any, Dict, List, Optional

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

    def run_query(self, query: str, **kwargs) -> Dict[str, str]:
        modalities = kwargs.get("modalities", [])
        if not modalities:
            raise ValueError("Modalities is empty.")

        self._logger.info(
            f'Running ParaphraseAgent with query: "{query}" and modalities: "{modalities}"'
        )
        result = super().run_query(query=query, modalities=modalities)
        self._logger.info(f'Paraphrase results: "{result}"')
        return result
