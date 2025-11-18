"""Entry point for running the offline OmniAnswer evaluation pipeline.

This script wires together content moderation, modality selection, paraphrasing,
and orchestration to process a list of learning queries, and optionally runs
the quantitative evaluation pipeline.
"""

import logging
import os

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm

from evaluation.evaluation_pipeline import EvaluationPipeline
from utils.general_utils import setup_logging
from utils.pipeline_utils import init_components, process_file


@hydra.main(version_base=None, config_path="../config", config_name="pipeline.yaml")
def main(cfg: DictConfig) -> None:
    """Run the end‑to‑end pipeline for a batch of questions.

    Args:
        cfg: Hydra configuration object containing model, agent, and
            evaluation settings.
    """
    load_dotenv()
    logger = logging.getLogger(__name__)
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "config", "logging.yaml"
        )
    )
    logger.info("Setting up logging configuration.")

    content_moderator, modality_agent, paraphrase_agent, orchestrator = init_components(
        cfg=cfg, logger=logger
    )

    queries = process_file(path=cfg.questions)

    for query in tqdm(queries):
        content_moderator.moderate_query(query=query)
        modalities = modality_agent.run_query(query=query)
        paraphrased_queries = paraphrase_agent.run_query(
            query=query, modalities=modalities
        )

        result_dict = orchestrator.run(
            query=query, paraphrase_queries=paraphrased_queries
        )

        if cfg.evaluate:
            evaluation_pipeline = EvaluationPipeline(
                cfg=cfg,
                logger=logger,
                result_dict=result_dict,
                llm_name=cfg.model,
                temperature=cfg.temperature,
            )
            evaluation_pipeline.evaluate()


if __name__ == "__main__":
    main()
