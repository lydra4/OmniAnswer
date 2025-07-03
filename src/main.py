import logging
import os

import hydra
import omegaconf
from agents.multi_modality.modality_agent import ModalityAgent
from agents.single_modality.image_agent import ImageAgent
from agents.single_modality.paraphrase_agent import ParaphraseAgent
from agents.single_modality.text_agent import TextAgent
from agents.single_modality.video_agent import VideoAgent
from teams.multi_modal_team import MultiModalTeam
from utils.general_utils import load_llm, setup_logging


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def main(cfg: omegaconf.DictConfig):
    logger = logging.getLogger(__name__)
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "config", "logging.yaml"
        )
    )
    logger.info("Setting up logging configuration.")

    query = "What does training a model on a GPU actually look like?"

    llm = load_llm(model_name=cfg.model, temperature=cfg.temperature)

    modality_agent = ModalityAgent(cfg=cfg, logger=logger, llm=llm)
    modalities = modality_agent.run(query=query)

    paraphrase_agent = ParaphraseAgent(cfg=cfg.paraphrase_agent, logger=logger, llm=llm)
    paraphrased_outputs = paraphrase_agent.run(query, modalities=modalities)

    multimodal_team = MultiModalTeam(
        text_agent=TextAgent(cfg=cfg, logger=logger, llm=llm),
        image_agent=ImageAgent(cfg=cfg, logger=logger, llm=llm),
        video_agent=VideoAgent(cfg=cfg, logger=logger, llm=llm),
        cfg=cfg.multimodal_team,
        logger=logger,
        llm=llm,
    )
    multimodal_team.run(query=query)


if __name__ == "__main__":
    main()
