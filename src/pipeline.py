import logging
import os

import hydra
from crewai import Crew
from dotenv import load_dotenv
from omegaconf import DictConfig

from agents.base_agent.base_agent_task import BaseAgentTask
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

    content_moderator = ContentModeratior(cfg=cfg, logger=logger)
    content_moderator.moderate_query(query=query)

    llm = load_llm(model_name=cfg.model, temperature=cfg.temperature)
    modality_agent = BaseAgentTask(
        cfg=cfg.modality_agent,
        logger=logger,
        llm=llm,
    )
    task = modality_agent.create_task(query=query)
    crew = Crew(agents=[modality_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    print(result)


if __name__ == "__main__":
    main()
