import logging
from typing import Any, List, Optional

from agents.base_agent import BaseAgent
from omegaconf import DictConfig
from tools.serpapi_search import SerpAPISearch


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
        tools: Optional[List[Any]] = None,
    ) -> None:
        """
        Initializes the TextAgent with configuration, logger, language model, and tools.

        Args:
            cfg (DictConfig): Hydra configuration object containing agent parameters.
            logger (logging.Logger): Logger instance for logging runtime information.
            llm: The language model to use for processing queries.
            tools (List[Any], optional): Custom list of tools to override defaults.
        """
        tools = [SerpAPISearch(cfg=cfg)] if tools is None else tools

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
        self.logger.info("Searching the web...")
        response = super().run(query)
        url = response.content.strip()

        if not url.startswith("http"):
            self.logger.warning(f"Invalid response: {url}.")

        self.logger.info(f"For text: {url}.")
        return url
