from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from agno.utils.log import logger
from jinja2 import Template
from omegaconf import DictConfig


class ParaphraseAgent(BaseAgent):
    """
    Agent for generating paraphrases of user queries tailored to specific learning modalities.

    This agent leverages a language model to create alternative versions of an input query
    suited to different modalities such as visual, auditory, or kinesthetic learning.
    """

    def __init__(
        self,
        cfg: DictConfig,
        llm,
        tools: Optional[List[Any]] = None,
    ) -> None:
        """
        Initializes the ParaphraseAgent with configuration, logger, and language model.

        Args:
            cfg (DictConfig): Hydra configuration object containing agent parameters.
            logger (logging.Logger): Logger instance for logging runtime info.
            llm: Language model instance used for paraphrasing queries.
            tools (List[Any], optional): Custom list of tools to override defaults.
        """
        super().__init__(cfg=cfg.paraphrase_agent, logger=logger, llm=llm, tools=tools)
        self.cfg = cfg
        self.raw_system_message = self.cfg.paraphrase_agent.system_message

    def _render_system_message(self, query: str, modality: str) -> str:
        """
        Render the system message using Jinja2 template, dynamically injecting modalities and example outputs.

        Args:
            modalities (List[str]): List of modalities to generate paraphrases for.

        Returns:
            str: Rendered system message prompt.
        """
        template = Template(self.raw_system_message)
        rendered_prompt = template.render(query=query, modality=modality)

        return rendered_prompt

    def run_query(self, query: str, **kwargs) -> Dict[str, str]:
        """
        Run the paraphrase agent to generate paraphrases for the given query.

        Args:
            query (str): The input query to be paraphrased.
            modalities (List[str]): List of modalities to consider for paraphrasing.
            **kwargs: Additional keyword arguments passed to the underlying LLM call.

        Returns:
            Dict[str, str]: A dictionary mapping each modality to its corresponding paraphrased query.

        Raises:
            ValueError: If the LLM response cannot be parsed into a valid Python dictionary.
        """
        modalities = kwargs.get("modalities", [])
        if not modalities:
            raise ValueError("Missing modalities in kwargs.")

        logger.info(
            f"Running ParaphraseAgent with query: {query} and modalities: {modalities}"
        )

        results: Dict[str, str] = {}

        for mode in modalities:
            system_prompt = self._render_system_message(query=query, modality=mode)

            try:
                response = super().run(
                    query=query, modality=mode, system_prompt=system_prompt, **kwargs
                )
                results[mode] = response.content.strip()

            except Exception as e:
                logger.error(f"Error processing modality {mode}: {e}")

        logger.info(f"Paraphrase results: {results}")
        return results
