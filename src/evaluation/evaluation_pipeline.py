import logging
from typing import Dict

from omegaconf import DictConfig



class EvaluationPipeline:
    def __init__(self, cfg: DictConfig, logger: logging.Logger):
        """
        Initializes the evaluation pipeline with configuration and logger.

        Args:
            cfg (DictConfig): Configuration for the evaluation pipeline.
            logger (logging.Logger): Logger instance for tracking execution.
        """
        self.cfg = cfg
        self.logger = logger

    def run(self, query: str, output: Dict[str, str]):
