import logging
import logging.config
import os
from typing import List, Optional

import yaml
from crewai import LLM

import mlflow

logger = logging.getLogger(__name__)


def setup_logging(
    logging_config_path: str = "../conf/logging.yaml", default_level: int = logging.INFO
) -> None:
    try:
        os.makedirs("logs", exist_ok=True)
        with open(logging_config_path, encoding="utf-8") as file:
            log_config = yaml.safe_load(file.read())
        logging.config.dictConfig(log_config)

    except Exception as error:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logger.info(error)
        logger.info("Logging config file is not found. Basic config is used.")


def load_llm(model_name: str, temperature: float) -> LLM:
    model_name_clean = model_name.strip().lower()

    if model_name_clean.startswith("gemini-"):
        model = f"gemini/{model_name_clean}"
        api_key = os.getenv("GEMINI_API_KEY")
    elif model_name_clean.startswith("gpt-"):
        model = f"openai/{model_name_clean}"
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Supported models are OpenAI and Gemini."
        )

    llm = LLM(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )
    logger.info(f"'{model_name}' loaded at temperature: '{temperature}'.")
    return llm


def init_mlflow(
    directory: str,
    experiment_name: str,
    llm_name: str,
    temperature: int | float,
    modes: List[str],
    text_similarity: Optional[float] = None,
    image_similarity: Optional[float] = None,
    video_similarity: Optional[float] = None,
) -> None:
    os.makedirs(name=directory, exist_ok=True)
    mlflow.set_tracking_uri(uri=f"file:{directory}")
    mlflow.set_experiment(experiment_name=experiment_name)

    metrics = {
        "Text Similarity": text_similarity,
        "Image Similarity": image_similarity,
        "Video Similarity": video_similarity,
    }

    run_name = f"{llm_name}-{temperature}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"llm": llm_name, "temperature": temperature})
        mlflow.set_tag("modes", ",".join(modes))
        for metric_name, value in metrics.items():
            if value is not None:
                mlflow.log_metric(metric_name, value)
