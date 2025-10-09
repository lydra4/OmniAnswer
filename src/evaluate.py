import logging
import os

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from evaluation.evaluation_pipeline import EvaluationPipeline
from utils.general_utils import setup_logging


@hydra.main(
    version_base=None, config_path="../config/evaluation", config_name="evaluation.yaml"
)
def main(cfg: DictConfig):
    load_dotenv()
    logger = logging.getLogger(__name__)
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "config", "logging.yaml"
        )
    )
    logger.info("Setting up logging configuration")

    evaluation_pipeline = EvaluationPipeline(cfg=cfg, logger=logger)
    evaluation_pipeline.evaluate()


if __name__ == "__main__":
    main()
