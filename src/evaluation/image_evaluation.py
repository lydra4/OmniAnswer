import logging
import os

from deepeval.models.llms import GeminiModel, GPTModel
from omegaconf import DictConfig


class ImageEvaluation:
    def __init__(
        self, cfg: DictConfig, logger: logging.Logger, query: str, output: str
    ):
        """
        Initializes the evaluation pipeline with configuration and logger.

        Args:
            cfg (DictConfig): Configuration for the evaluation pipeline.
            logger (logging.Logger): Logger instance for tracking execution.
        """
        self.cfg = cfg
        self.logger = logger
        self.query = query
        self.output = output
        self.llm = (
            GPTModel(model=self.cfg.model, api_key=os.getenv("OPENAI_API_KEY"))
            if self.cfg.model.startswith("gpt-")
            else GeminiModel(
                model_name=self.cfg.model, api_key=os.getenv("GEMINI_API_KEY")
            )
        )
