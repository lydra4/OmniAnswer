import logging
import os
from typing import Union

import yaml
from crewai import LLM

logger = logging.getLogger(__name__)


def setup_logging(
    logging_config_path="../conf/logging.yaml", default_level=logging.INFO
):
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
