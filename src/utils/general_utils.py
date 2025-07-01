import ast
import logging
import logging.config
import os
import re
from typing import List, Tuple, Union

import yaml
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def setup_logging(
    logging_config_path="../conf/logging.yaml", default_level=logging.INFO
):
    """
    Initializes the logging configuration.

    Attempts to load logging settings from a YAML configuration file. If the file is missing or malformed,
    it falls back to a basic logging setup.

    Args:
        logging_config_path (str): Path to the logging configuration YAML file.
        default_level (int): Logging level to use if YAML config is not found or fails to load.

    Raises:
        Any exception during YAML reading or logging setup will be logged and ignored, using fallback logging.
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
    Loads a large language model (LLM) based on the model name.

    Args:
        model_name (str): Name of the model to load. Should start with 'gpt-' or 'gemini-'.
        temperature (Union[int, float]): Temperature setting to control response randomness.

    Returns:
        Union[OpenAIChat, Gemini]: Instantiated LLM object.

    Raises:
        ValueError: If the model name does not match supported providers.
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


def extract_image_urls(text: str) -> List[str]:
    """
    Extract image URLs from markdown or plain text.

    Supports:
    - Markdown format: ![alt](https://example.com)
    - Plain bullet list: * https://example.com

    Args:
        text (str): Input text containing image URLs.

    Returns:
        List[str]: List of valid image URLs.
    """
    # Match markdown-style and bare URLs
    urls = re.findall(r"\(?(https?://[^\s)]+)\)?", text)
    return [url.strip(").,") for url in urls]


def extract_video_titles_and_urls(text: str) -> List[Tuple[str, str]]:
    """
    Extracts video titles and URLs from formatted markdown text.

    Args:
        text (str): The text containing video titles in `**Title**` format and associated URLs.

    Returns:
        List[Tuple[str, str]]: A list of tuples with (video title, video URL).
    """
    return re.findall(r"\*\*([^\*]+)\*\*[\s\S]*?URL:\s*(https?://[^\s]+)", text)


def extract_python_json_block(text: str) -> List[str]:
    """
    Extracts a Python or JSON code block from the text.

    Args:
        text (str): The input text containing code blocks.

    Returns:
        Union[str, None]: The extracted code block as a string, or None if no block is found.
    """
    cleaned = re.sub(
        r"^```(?:json|python)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    return ast.literal_eval(cleaned)
