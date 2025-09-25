import logging
import os

import hydra
from crewai import Agent, Crew, Task
from crewai_tools import TavilySearchTool
from dotenv import load_dotenv
from omegaconf import DictConfig

from agents.modality_agent import ModalityAgent
from agents.paraphrase_agent import ParaphraseAgent
from moderation.content_moderator import ContentModeratior
from schemas.schemas import DictOutput, StringListOutput
from tools.image_search import image_search
from tools.video_search import video_search
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
    paraphrased_queries["video"] = (
        "Tutorial on Model Context Protocol for agentic workflows"
    )

    tasks = []
    agents = []
    if "text" in paraphrased_queries:
        text_agent = Agent(
            llm=llm,
            tools=[
                TavilySearchTool(
                    api_key=os.getenv("TAVILY_API_KEY"),
                    max_results=cfg.text_agent.max_results,
                    include_images=False,
                    exclude_domains=["youtube.com", "youtu.be"],
                )
            ],
            **cfg.text_agent.agent,
        )
        agents.append(text_agent)
        tasks.append(
            Task(
                description=paraphrased_queries["text"],
                agent=text_agent,
                expected_output=cfg.text_agent.task.expected_output,
            )
        )

    if "image" in paraphrased_queries:
        image_agent = Agent(llm=llm, tools=[image_search], **cfg.image_agent.agent)
        agents.append(image_agent)
        tasks.append(
            Task(
                description=paraphrased_queries["image"],
                agent=image_agent,
                expected_output=cfg.image_agent.task.expected_output,
            )
        )

    if "video" in paraphrased_queries:
        video_agent = Agent(llm=llm, tools=[video_search], **cfg.video_agent.agent)
        agents.append(video_agent)
        tasks.append(
            Task(
                description=paraphrased_queries["video"],
                agent=video_agent,
                expected_output=cfg.video_agent.task.expected_output,
            )
        )

    crew = Crew(agents=agents, tasks=tasks, verbose=False)
    result = crew.kickoff()
    print(result)


if __name__ == "__main__":
    main()
