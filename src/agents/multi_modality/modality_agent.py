import logging
from typing import Any, List

import nltk
from agents.base_agent import BaseAgent
from agno.tools.reasoning import ReasoningTools
from guardrails.guard import Guard
from guardrails.hub import BanList, ToxicLanguage
from omegaconf import DictConfig
from utils.general_utils import extract_python_json_block

nltk.download("punkt")


class ModalityAgent(BaseAgent):
    """
    Agent that determines the best learning modalities for a given user query.

    It uses guardrails to reject queries containing banned words or toxic language before processing
    with a language model to return modality suggestions.
    """

    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
        tools: List[Any] = None,
    ) -> None:
        """
        Initializes the ModalityAgent with configuration, logger, LLM, and validation guardrails.

        Args:
            cfg (DictConfig): Configuration specific to the modality agent.
            logger (logging.Logger): Logger instance for tracking execution.
            llm (Any): The language model used to generate responses.
        """
        tools = [ReasoningTools()] if tools is None else tools
        super().__init__(cfg=cfg.modality_agent, logger=logger, llm=llm, tools=tools)
        self.guard = Guard().use_many(
            BanList(
                banned_words=cfg.modality_agent.guardrails.banned_words,
                on_fail="refrain",
            ),
            ToxicLanguage(
                threshold=cfg.modality_agent.guardrails.toxic_threshold,
                validation_method="sentence",
                on_fail="refrain",
            ),
        )

    def run(self, query: str) -> List[str]:
        """
        Validates and processes the user query to determine optimal learning modalities.

        Applies language safety checks, and if passed, uses the LLM to suggest modalities
        such as visual, auditory, kinesthetic, etc.

        Args:
            query (str): User input about their learning preferences or needs.

        Returns:
            List[str]: A list of learning modalities derived from the query.

        Raises:
            ValueError: If the query fails any validation guardrails.
        """
        self.logger.info(f'Running on query: "{query}".')
        result = self.guard.validate(query)

        if not result.validation_passed:
            reasons_by_validator = {}

            for summary in result.validation_summaries:
                if summary.validator_status == "fail":
                    all_reasons = [span.reason for span in summary.error_spans]
                    reasons_by_validator[summary.validator_name] = all_reasons

                    for reason in all_reasons:
                        self.logger.warning(
                            f"{summary.validator_name} failed: {reason}"
                        )

            formatted = []
            for validator, reasons in reasons_by_validator.items():
                for reason in reasons:
                    formatted.append(f"{validator}: {reason}")

            raise ValueError("Rejected query due to:\n" + "\n".join(formatted))

        response = super().run(query=query)
        modalities = extract_python_json_block(response.content.strip())
        self.logger.info(
            f'For the query:"{query}", best modes of learning: {modalities}.'
        )
        return modalities
