from typing import Any, List, Optional

from agents.base_agent import BaseAgent
from agno.utils.log import logger
from omegaconf import DictConfig
from tools.text_search import TextSearch


class TextAgent(BaseAgent):
    def __init__(
        self,
        cfg: DictConfig,
        llm,
        tools: Optional[List[Any]] = None,
    ) -> None:
        tools = (
            [
                TextSearch(cfg=cfg),
            ]
            if tools is None
            else tools
        )

        super().__init__(cfg=cfg.text_agent, logger=logger, llm=llm, tools=tools)

    def run_query(self, query: str, **kwargs):
        response = super().run(message=query)
        url = response.content.strip()

        if not url.startswith("http"):
            logger.warning(f"Invalid response: {url}.")

        logger.info(f"For text: {url}.")
        return url
