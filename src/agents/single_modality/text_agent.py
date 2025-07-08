import ast
import logging
from typing import Any, List

from agents.base_agent import BaseAgent
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.reasoning import ReasoningTools
from omegaconf import DictConfig


class TextAgent(BaseAgent):
    """
    Agent for searching and returning relevant text-based documents from the web.

    This agent uses Google Search via Agno's toolchain to find up-to-date, relevant articles,
    documents, or web pages based on user queries.
    """

    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
        tools: List[Any] = None,
    ) -> None:
        """
        Initializes the TextAgent with configuration, logger, language model, and tools.

        Args:
            cfg (DictConfig): Hydra configuration object containing agent parameters.
            logger (logging.Logger): Logger instance for logging runtime information.
            llm: The language model to use for processing queries.
            tools (List[Any], optional): Custom list of tools to override defaults.
        """
        tools = (
            [
                ReasoningTools(),
                GoogleSearchTools(
                    stop_after_tool_call_tools=["google_search"],
                    show_result_tools=["google_search"],
                    fixed_max_results=cfg.text_agent.fixed_max_results,
                ),
            ]
            if tools is None
            else tools
        )

        super().__init__(cfg=cfg.text_agent, logger=logger, llm=llm, tools=tools)

    def run(self, query: str, **kwargs):
        """
        Run the TextAgent to process the given query.

        Args:
            query (str): The input query string used to search for relevant documents.

        Returns:
            List[Dict[str, str]]: A list of dictionaries, each containing:
                - title (str): Title of the retrieved document.
                - url (str): URL link to the document.

        Raises:
            ValueError: If the LLM response cannot be parsed into the expected list of dicts.
        """
        self.logger.info(f"Looking up text documents with query: {query}.")
        response = super().run(query)
        result_list = ast.literal_eval(response.content)

        if not result_list:
            self.logger.warning("No text results found.")
            return "No text results found."

        url_result = [item["url"] for item in result_list]
        url = " ".join(url_result)
        self.logger.info(f"For text: {[url]}.")

        return url
