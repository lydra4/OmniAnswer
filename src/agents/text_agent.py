import ast
import logging
from typing import Any, List

from agents.base_agent import BaseAgent
from agno.tools.googlesearch import GoogleSearchTools
from omegaconf import DictConfig


class TextAgent(BaseAgent):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
        tools: List[Any] = None,
    ) -> None:
        tools = (
            [
                GoogleSearchTools(
                    stop_after_tool_call_tools=["google_search"],
                    show_result_tools=["google_search"],
                    fixed_max_results=cfg.text_agent.fixed_max_results,
                )
            ]
            if tools is None
            else tools
        )

        super().__init__(cfg=cfg.text_agent, logger=logger, llm=llm, tools=tools)

    def run(self, query: str):
        """
        Run the TextAgent to process the given query.

        Args:
            query (str): The input query to process.

        Returns:
            List[str]: A list of responses generated by the agent.
        """
        self.logger.info(f"Looking up text documents with query: {query}.")
        response = super().run(query)
        result = ast.literal_eval(response.content)
        result_dict = [
            {"title": dictionary["title"], "url": dictionary["url"]}
            for dictionary in result
        ]
        self.logger.info(f"URL of text documents: {result_dict}.")
        return result_dict
