import logging
import os

import hydra
import omegaconf
from agents.multi_modality.modality_agent import ModalityAgent
from agents.single_modality.paraphrase_agent import ParaphraseAgent
from evaluation.evaluation_pipeline import EvaluationPipeline
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

    paraphrase_agent = ParaphraseAgent(cfg=cfg, logger=logger, llm=llm)
    paraphrased_outputs = paraphrase_agent.run(query, modalities=modalities)

    multimodal_team = MultiModalTeam(
        paraphrased_outputs=paraphrased_outputs, cfg=cfg, logger=logger, llm=llm
    )
    output = multimodal_team.run(query=query)

    evaluation_pipeline = EvaluationPipeline(
        cfg=cfg, logger=logger, query=query, output=output
    )
    evaluation_pipeline.evaluate_text_agent(text_output=output["text"])


if __name__ == "__main__":
    main()
