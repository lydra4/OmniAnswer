import logging
import os
from abc import ABC, abstractmethod
from typing import Dict

from deepeval.models.llms import GeminiModel, GPTModel
from omegaconf import DictConfig


class BaseEvaluation(ABC):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        query: str,
        output: Dict[str, str],
    ):
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

    @abstractmethod
    def evaluate_all(self) -> Dict[str, float]:
        pass
