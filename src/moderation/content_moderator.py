import logging
import os

from omegaconf import DictConfig
from openai import OpenAI


class ContentModeratior:
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
    ):
        self.cfg = cfg
        self.logger = logger
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def moderate_query(self, query: str) -> None:
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
