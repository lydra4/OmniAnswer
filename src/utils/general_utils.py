import ast
import logging
import logging.config
import os
import re
from typing import List, Union

import yaml
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv

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
    load_dotenv()
    model_id = model_name.strip().lower()

    if model_id.startswith("gpt-"):
        return OpenAIChat(
            id=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
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
    urls = re.findall(r"\(?(https?://[^\s)]+)\)?", text)
    return [url.strip(").,") for url in urls]


def extract_video_urls(text: str) -> List[str]:
    return re.findall(r"https?://www\.youtube\.com/watch\?v=[\w-]+", text)


def extract_python_json_block(text: str) -> List[str]:
    cleaned = re.sub(
        r"^```(?:json|python)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    return ast.literal_eval(cleaned)
