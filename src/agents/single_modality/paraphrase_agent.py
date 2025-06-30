import ast
import logging
import re
from typing import Dict, List

from agents.base.base_agent import BaseAgent
from omegaconf import DictConfig


class ParaphraseAgent(BaseAgent):
    """
    Agent for generating paraphrases of user queries tailored to specific learning modalities.

    This agent leverages a language model to create alternative versions of an input query
    suited to different modalities such as visual, auditory, or kinesthetic learning.
    """

    def __init__(self, cfg: DictConfig, logger: logging.Logger, llm) -> None:
        """
        Initializes the ParaphraseAgent with configuration, logger, and language model.

        Args:
            cfg (DictConfig): Hydra configuration object containing agent parameters.
            logger (logging.Logger): Logger instance for logging runtime info.
            llm: Language model instance used for paraphrasing queries.
        """
        super().__init__(cfg=cfg.paraphrase_agent, logger=logger, llm=llm)

    def run(self, query: str, modalities: List[str], **kwargs) -> Dict[str, str]:
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
        self.logger.info(f"Running Paraphrase Agent with query: {query}.")
        response = super().run(query=query, modalities=modalities, **kwargs)
        cleaned = re.sub(
            r"^```(?:json|python)?\s*|\s*```$",
            "",
            response.content.strip(),
            flags=re.IGNORECASE,
        )
        paraphrased_outputs = ast.literal_eval(cleaned)
        self.logger.info(
            f"The different phrase(s) for the different mode(s): {paraphrased_outputs}."
        )
        return paraphrased_outputs
