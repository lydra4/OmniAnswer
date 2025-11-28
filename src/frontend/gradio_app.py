"""Gradio-based web frontend for interacting with OmniAnswer."""

import logging
from typing import List, Optional, Tuple

import gradio as gr
import omegaconf

from utils.pipeline_utils import init_components


class GradioApp:
    """Wrapper around the Gradio UI and backend pipeline components."""

    def __init__(
        self, cfg: omegaconf.DictConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        """Initialize the Gradio application.

        Args:
            cfg: Hydra/OMEGACONF configuration for the pipeline and UI.
            logger: Optional logger instance; if omitted, a module-level logger
                is created.
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        (
            self.content_moderator,
            self.modality_agent,
            self.paraphrase_agent,
            self.orchestrator,
        ) = init_components(cfg=self.cfg, logger=self.logger)
        self.caption = self.cfg.gradio.caption

    def _obtain_modes(self, query: str) -> Tuple[str, str, List[str]]:
        """Determine the best modalities for a query and craft a user message.

        Args:
            query: Raw user query from the chat interface.

        Returns:
            A tuple of ``(message, query, modalities)`` where ``message`` is the
            explanatory text describing the chosen modalities.
        """
        self.content_moderator.moderate_query(query=query)
        modalities = self.modality_agent.run_query(query=query)
        n = len(modalities)
        if n == 1:
            mode_str = modalities[0]
            msg = f"For {query}, {mode_str} is the best mode to learn."
        elif n == 2:
            mode_str = f"{modalities[0]} and {modalities[1]}"
            msg = f"For {query}, {mode_str} are the best mode to learn."
        else:
            mode_str = ", ".join(modalities[:-1]) + " and " + modalities[-1]
            msg = f"For '**{query}**', '**{mode_str}**' are the best mode to learn."

        return msg, query, modalities

    async def _obtain_urls(self, query: str, modalities: List[str]) -> str:
        """Produce URLs for each modality given a query.

        Args:
            query: User query to answer.
            modalities: Modalities selected for this query.

        Returns:
            A human-readable string listing the URLs for each modality.
        """
        paraphrased_queries = self.paraphrase_agent.run_query(
            query=query, modalities=modalities
        )
        result_dict = await self.orchestrator.run(
            query=query, paraphrase_queries=paraphrased_queries
        )
        results_list = result_dict["results"]
        result_text = "\n".join(
            [
                f"For {mode_dict['modality']}: {mode_dict['url']}"
                for mode_dict in results_list
            ]
        )
        return result_text

    async def _infer(self, query: str, chat_history: List[Tuple[str, str]]):
        """Streaming inference callback used by the Gradio Chatbot.

        Args:
            query: Current user message.
            chat_history: Existing chat history between user and assistant.

        Yields:
            Updated chat history and an (unused) text value for the input box.
        """
        chat_history = chat_history + [(query, "")]
        yield chat_history, ""

        msg, query, modalities = self._obtain_modes(query=query)
        chat_history[-1] = (query, msg)
        yield chat_history, ""
        result_text = await self._obtain_urls(query=query, modalities=modalities)
        chat_history[-1] = (
            chat_history[-1][0],
            chat_history[-1][1] + "\n\n" + result_text,
        )
        yield chat_history, ""

    def launch_app(self):
        """Create and launch the Gradio Blocks interface."""
        with gr.Blocks() as demo:
            chatbot = gr.Chatbot()
            query = gr.Textbox(placeholder=self.caption)
            send = gr.Button("Send")

            send.click(
                fn=self._infer, inputs=[query, chatbot], outputs=[chatbot, query]
            )
            query.submit(
                fn=self._infer, inputs=[query, chatbot], outputs=[chatbot, query]
            )
        demo.launch(server_name="0.0.0.0", server_port=8080)
