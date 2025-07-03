import logging
from typing import Any, Dict, List

from agents.base.base_agent import BaseAgent
from agno.tools.thinking import ThinkingTools
from jinja2 import Template
from omegaconf import DictConfig
from utils.general_utils import extract_python_json_block


class ParaphraseAgent(BaseAgent):
    """
    Agent for generating paraphrases of user queries tailored to specific learning modalities.

    This agent leverages a language model to create alternative versions of an input query
    suited to different modalities such as visual, auditory, or kinesthetic learning.
    """

    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        llm,
        tools: List[Any] = None,
    ) -> None:
        """
        Initializes the ParaphraseAgent with configuration, logger, and language model.

        Args:
            cfg (DictConfig): Hydra configuration object containing agent parameters.
            logger (logging.Logger): Logger instance for logging runtime info.
            llm: Language model instance used for paraphrasing queries.
            tools (List[Any], optional): Custom list of tools to override defaults.
        """
        tools = [ThinkingTools()] if tools is None else tools
        super().__init__(cfg=cfg, logger=logger, llm=llm, tools=tools)
        self.raw_system_message = self.cfg.system_message

    def _render_system_message(self, modalities: List[str]) -> str:
        """
        Render the system message using Jinja2 template, dynamically injecting modalities and example outputs.

        Args:
            modalities (List[str]): List of modalities to generate paraphrases for.

        Returns:
            str: Rendered system message prompt.
        """
        example_dict = {m: f"paraphrase for {m}" for m in modalities}
        example_str = str(example_dict).replace("'", '"')
        template = Template(self.raw_system_message)
        rendered_prompt = template.render(modalities=modalities, example=example_str)

        return rendered_prompt

    def run(self, query: str, modalities: List[str], **kwargs) -> Dict[str, str]:
        """
        Run the paraphrase agent to generate paraphrases for the given query.

        Args:
            query (str): The input query to be paraphrased.
            modalities (List[str]): List of modalities to consider for paraphrasing.
            **kwargs: Additional keyword arguments passed to the underlying LLM call.

        Returns:
            Dict[str, str]: A dictionary mapping each modality to its corresponding paraphrased query.

        Raises:
            ValueError: If the LLM response cannot be parsed into a valid Python dictionary.
        """
        self.logger.info(
            f"Running ParaphraseAgent with query: {query} and modalities: {modalities}"
        )
        system_prompt = self._render_system_message(modalities=modalities)
        response = super().run(
            query=query, modalities=modalities, system_prompt=system_prompt, **kwargs
        )
        paraphrased_outputs = extract_python_json_block(text=response.content.strip())
        self.logger.info(
            f"The different phrase(s) for the different mode(s): {paraphrased_outputs}."
        )
        return paraphrased_outputs
