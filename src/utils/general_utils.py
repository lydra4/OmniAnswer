"""General utility functions for logging, model loading, and MLflow setup."""

import logging
import logging.config
import os
from typing import List, Optional

import mlflow
import yaml
from crewai import LLM

logger = logging.getLogger(__name__)


def setup_logging(
    logging_config_path: str = "../conf/logging.yaml", default_level: int = logging.INFO
) -> None:
    """Configure application-wide logging using a YAML config file.

    If the configuration file is missing or invalid, a basic logging
    configuration is installed instead.

    Args:
        logging_config_path: Path to the logging YAML configuration file.
        default_level: Logging level used for the fallback basic configuration.
    """
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
    """Instantiate a CrewAI LLM wrapper for the requested model.

    The function currently supports OpenAI (``gpt-``) and Gemini (``gemini-``)
    model families, using the appropriate API keys from the environment.

    Args:
        model_name: Name of the model, e.g. ``\"gpt-4o\"`` or ``\"gemini-2.0-pro\"``.
        temperature: Sampling temperature for the model.

    Returns:
        A configured :class:`~crewai.LLM` instance.

    Raises:
        ValueError: If the model family is not supported or no API key is set.
    """
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
    """Initialize an MLflow experiment and log evaluation metrics.

    Args:
        directory: Local directory used as the MLflow tracking URI.
        experiment_name: Name of the MLflow experiment.
        llm_name: Identifier of the language model under evaluation.
        temperature: Sampling temperature used with the language model.
        modes: List of modalities included in the evaluation.
        text_similarity: Optional similarity score for text.
        image_similarity: Optional similarity score for images.
        video_similarity: Optional similarity score for video.
    """
    os.makedirs(name=directory, exist_ok=True)
    mlflow.set_tracking_uri(uri=f"file:{directory}")  # type: ignore[attr-defined]
    mlflow.set_experiment(experiment_name=experiment_name)  # type: ignore[attr-defined]

    metrics = {
        "Text Similarity": text_similarity,
        "Image Similarity": image_similarity,
        "Video Similarity": video_similarity,
    }

    run_name = f"{llm_name}-{temperature}"
    with mlflow.start_run(run_name=run_name):  # type: ignore[attr-defined]
        mlflow.log_params({"llm": llm_name, "temperature": temperature})  # type: ignore[attr-defined]
        mlflow.set_tag("modes", ",".join(modes))  # type: ignore[attr-defined]
        for metric_name, value in metrics.items():
            if value is not None:
                mlflow.log_metric(metric_name, value)  # type: ignore[attr-defined]
