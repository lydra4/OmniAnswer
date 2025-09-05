import logging
import os

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from agents.image_agent import ImageAgent
from agents.modality_agent import ModalityAgent
from agents.paraphrase_agent import ParaphraseAgent
from agents.text_agent import TextAgent
from agents.video_agent import VideoAgent
from moderation.content_moderator import ContentModeratior
from utils.general_utils import load_llm, setup_logging


@hydra.main(version_base=None, config_path="../config", config_name="pipeline.yaml")
def main(cfg: DictConfig):
    load_dotenv()
    logger = logging.getLogger(__name__)
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "config", "logging.yaml"
        )
    )
    logger.info("Setting up logging configuration.")

    query = "Please explain model context protocol in the context of agentic workflow."

    llm = load_llm(model_name=cfg.model, temperature=cfg.temperature)

    content_moderator = ContentModeratior(cfg=cfg, logger=logger)
    content_moderator.moderate_query(query=query)

    modality_agent = ModalityAgent(cfg=cfg, logger=logger, llm=llm)
    modalities = modality_agent.run_query(query=query)

    paraphrase_agent = ParaphraseAgent(cfg=cfg, logger=logger, llm=llm)
    paraphrased_outputs = paraphrase_agent.run_query(query=query, modalities=modalities)
    paraphrased_modalities = list(paraphrased_outputs.keys())

    if "text" in paraphrased_modalities:
        text_agent = TextAgent(cfg=cfg, logger=logger, llm=llm)
        # url = text_agent.run_query(query=query)

    if "image" in paraphrased_modalities:
        image_agent = ImageAgent(cfg=cfg, logger=logger, llm=llm)
        url = image_agent.run_query(query=query)

    if "video" in paraphrased_modalities:
        video_agent = VideoAgent(cfg=cfg, logger=logger, llm=llm)
        # url = video_agent.run_query(query=query)


if __name__ == "__main__":
    main()
