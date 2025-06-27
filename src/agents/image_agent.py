import logging
from typing import Any, List

from agents.base_agent import BaseAgent
from agno.tools.duckduckgo import DuckDuckGoTools
from omegaconf import DictConfig


class ImageAgent(BaseAgent):
    def __init__(
        self, cfg: DictConfig, logger: logging.Logger, llm, tools: List[Any] = None
    ):
        if tools is None:
            tools = [
                DuckDuckGoTools(
                    stop_after_tool_call_tools=["duckduckgo_image"],
                    show_result_tools=["duckduckgo_image"],
                    fixed_max_results=cfg.image_agent.fixed_max_results,
                )
            ]
        super().__init__(cfg=cfg.image_agent, logger=logger, llm=llm, tools=tools)

    def run(self, query: str):
        """
        Run the ImageAgent to process the given query.

        Args:
            query (str): The input query to process.

        Returns:
            List[dict]: A list of dictionaries containing image titles and URLs.
        """
        self.logger.info(f"Running ImageAgent with query: {query}.")
        response = super().run(query)
        return response
