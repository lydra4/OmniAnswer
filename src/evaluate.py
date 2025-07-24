import json
import logging
import os

import hydra
from evaluation.image_evaluation import ImageEvaluation
from evaluation.text_evaluation import TextEvaluation
from omegaconf import DictConfig
from tqdm import tqdm
from utils.general_utils import setup_logging


@hydra.main(version_base=None, config_path="../config", config_name="evaluation.yaml")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "config", "logging.yaml"
        )
    )
    logger.info("Setting up logging configuration")

    with open(
        cfg.output_path,
        encoding="utf-8",
    ) as f:
        multimodal_output = json.load(f)

    text_evaluation = TextEvaluation(cfg=cfg, logger=logger)
    image_evaluation = ImageEvaluation(cfg=cfg, logger=logger)

    for output_data in tqdm(multimodal_output):
        query = output_data["original_query"]
        modalities = list(output_data["results"].keys())

        if "text" in modalities:
            text_evaluation.evaluate_all(query=query, output_data=output_data)
        if "image" in modalities:
            image_evaluation.evaluate_all(query=query, output_data=output_data)


if __name__ == "__main__":
    main()
