import logging
import os

import hydra
import omegaconf
from agents.modality_agent import ModalityAgent
from agents.paraphrase_agent import ParaphraseAgent
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

    paraphrase_agent = ParaphraseAgent(cfg=cfg, logger=logger, llm=llm)
    paraphrased_outputs = paraphrase_agent.run(query, modalities=modalities)

    print(paraphrased_outputs)


if __name__ == "__main__":
    main()
