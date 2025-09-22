import logging
import os

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from agents.image_agent import ImageAgent
from agents.modality_agent import ModalityAgent
from agents.paraphrase_agent import ParaphraseAgent
from agents.text_agent import TextAgent
from moderation.content_moderator import ContentModeratior
from schemas.schemas import DictOutput, StringListOutput, StringOutput
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

    content_moderator = ContentModeratior(cfg=cfg, logger=logger)
    content_moderator.moderate_query(query=query)

    llm = load_llm(model_name=cfg.model, temperature=cfg.temperature)
    modality_agent = ModalityAgent(
        cfg=cfg.modality_agent,
        logger=logger,
        llm=llm,
        output=StringListOutput,
    )
    modalities = modality_agent.run_query(query=query)

    paraphrase_agent = ParaphraseAgent(
        cfg=cfg.paraphrase_agent,
        logger=logger,
        llm=llm,
        output=DictOutput,
    )
    paraphrased_queries = paraphrase_agent.run_query(query=query, modalities=modalities)

    if "text" in paraphrased_queries:
        text_agent = TextAgent(
            cfg=cfg.text_agent,
            logger=logger,
            llm=llm,
            output=StringOutput,
        )
        text_result = text_agent.run_query(query=paraphrased_queries["text"])

    if "image" in paraphrased_queries:
        image_agent = ImageAgent(
            cfg=cfg.image_agent,
            logger=logger,
            llm=llm,
            output=StringOutput,
        )
        image_result = image_agent.run_query(query=paraphrased_queries["image"])


if __name__ == "__main__":
    main()
