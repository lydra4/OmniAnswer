"""CLI entry point for launching the OmniAnswer Gradio web application."""

import logging
import os

import hydra
from dotenv import load_dotenv

from frontend.gradio_app import GradioApp
from utils.general_utils import setup_logging


@hydra.main(version_base=None, config_path="../config", config_name="pipeline.yaml")
def main(cfg):
    """Hydra-wrapped main function that configures and launches Gradio."""
    load_dotenv()
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "config", "logging.yaml"
        )
    )

    gradio_app = GradioApp(
        cfg=cfg,
        logger=logger,
    )
    gradio_app.launch_app()


if __name__ == "__main__":
    main()
