import logging
import os

import hydra
import omegaconf
from agents.modality_classifier import ModalityClassifier
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
    logger.info(f"Using {cfg.model}.")

    modalityclassifier = ModalityClassifier(cfg=cfg, logger=logger)
    response = modalityclassifier.run(
        query="Show how the Model Context Protocol works, with visuals and a demo video of it in action."
    )
    print(response)


if __name__ == "__main__":
    main()
