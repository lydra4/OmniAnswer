import logging
from typing import List, Tuple

from omegaconf import DictConfig

from agents.modality_agent import ModalityAgent
from agents.paraphrase_agent import ParaphraseAgent
from crew.orchestrator import Orchestrator
from moderation.content_moderator import ContentModeratior
from schemas.schemas import DictOutput, StringListOutput
from utils.general_utils import load_llm


def init_components(
    cfg: DictConfig,
    logger: logging.Logger,
) -> Tuple[
    ContentModeratior,
    ModalityAgent,
    ParaphraseAgent,
    Orchestrator,
]:
    llm = load_llm(model_name=cfg.model, temperature=cfg.temperature)
    content_moderator = ContentModeratior(cfg=cfg, logger=logger)
    modality_agent = ModalityAgent(
        cfg=cfg.modality_agent, logger=logger, llm=llm, output=StringListOutput
    )
    paraphrase_agent = ParaphraseAgent(
        cfg=cfg.paraphrase_agent, logger=logger, llm=llm, output=DictOutput
    )
    orchestrator = Orchestrator(cfg=cfg, logger=logger, llm=llm)

    return content_moderator, modality_agent, paraphrase_agent, orchestrator


def process_file(path: str) -> List[str]:
    with open(file=path, mode="r", encoding="utf-8") as questions:
        queries = [question.strip() for question in questions]
    return queries
