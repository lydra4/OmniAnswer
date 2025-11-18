"""Content moderation helper built on top of the OpenAI Moderations API."""

import logging
import os

from omegaconf import DictConfig
from openai import OpenAI


class ContentModeratior:
    """Moderate incoming user queries using OpenAI safety models."""

    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
    ):
        """Initialize the content moderator.

        Args:
            cfg: Configuration containing the moderation model name.
            logger: Logger instance used for moderation decisions.
        """
        self.cfg = cfg
        self.logger = logger
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def moderate_query(self, query: str) -> None:
        """Validate that a query passes content safety checks.

        Args:
            query: Raw user query text to be moderated.

        Raises:
            ValueError: If any unsafe content categories are triggered.
        """
        self.logger.info(f"Moderating query:'{query}'.")
        response = self.client.moderations.create(
            model=self.cfg.moderation_model,
            input=query,
        ).model_dump()
        [results] = response["results"]
        true_categories = [
            key for key, value in results["categories"].items() if value is True
        ]
        if true_categories:
            self.logger.error(f"Rejected query due to {' and '.join(true_categories)}.")
            raise ValueError(f"Rejected query due to {' and '.join(true_categories)}.")
        else:
            self.logger.info(f"Query:'{query}' passed.")
