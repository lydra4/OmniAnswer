import logging
import logging.config
import os

import yaml
from crewai.project import llm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI

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


@llm
def load_llm(model_name: str, temperature: float):
    model = model_name.strip().lower()
    if model.startswith("gemini-"):
        return ChatGoogleGenerativeAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            model=model,
            temperature=temperature,
        )

    elif model.startswith("gpt-"):
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model,
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Supported models are OpenAI and Gemini."
        )
