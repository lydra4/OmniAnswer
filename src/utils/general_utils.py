import logging
import logging.config
import os
from typing import Union

import yaml
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def setup_logging(
    logging_config_path="../conf/logging.yaml", default_level=logging.INFO
):
    """
    Logging configuration module.

    This module provides functionality to set up logging using a YAML configuration file.
    If the configuration file is missing or invalid, it defaults to basic logging with a specified level.

    Attributes:
        logger (logging.Logger): Logger used to capture logs during setup.

    Functions:
        setup_logging(logging_config_path, default_level): Initializes logging from YAML or falls back to basic config.
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


def load_llm(model_name: str, temperature: Union[int, float]):
    """
    Load the appropriate LLM based on the model name.
    Args:
        model_name (str): The name of the model to load.
        temperature (int, float): The temperature setting for the model.

    Returns:
        An instance of OpenAIChat or Gemini based on the model name.
    Raises:
        ValueError: If the model name does not match any supported models.
    """
    load_dotenv()
    model_id = model_name.strip().lower()

    if model_id.startswith("gpt-"):
        return OpenAIChat(
            id=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif model_id.startswith("gemini-"):
        return Gemini(
            id=model_name,
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Supported models are OpenAI and Gemini."
        )
