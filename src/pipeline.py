import logging
import os

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from agents.modality_agent import ModalityAgent
from agents.paraphrase_agent import ParaphraseAgent
from crew.orchestrator import Orchestrator
from moderation.content_moderator import ContentModeratior
from schemas.schemas import DictOutput, StringListOutput
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

    orchestrator = Orchestrator(cfg=cfg, logger=logger, llm=llm)
    result = orchestrator.run(paraphrase_queries=paraphrased_queries)
    print(result)


if __name__ == "__main__":
    main()
