import logging
import os

import hydra
import omegaconf
from utils.general_utils import setup_logging


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def main(cfg: omegaconf.DictConfig):
    logger = logging.getLogger(__name__)
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "config", "logging.yaml"
        )
    )
    logger.info("Setting up logging configuration.")
    logger.info(f"Using {cfg.llm_model}.")


if __name__ == "__main__":
    main()
