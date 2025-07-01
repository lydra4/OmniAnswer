import logging
from typing import Any, List

from omegaconf import DictConfig


class MultiModalTeam:
    def __init__(
        self, cfg: DictConfig, logger: logging.Logger, llm, tools: List[Any] = None
    ):
        self.cfg = cfg
        self.logger = logger
        self.llm = llm
        self.tools = tools

    def run(self, query: str):
        self.logger.info(f"Running MultiModalTeam with query: {query}")
