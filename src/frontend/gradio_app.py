import logging
from typing import Generator, List, Optional, Tuple

import gradio as gr
import omegaconf

from utils.pipeline_utils import init_components


class GradioApp:
    def __init__(
        self, cfg: omegaconf.DictConfig, logger: Optional[logging.Logger] = None
    ) -> None:
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

    def _obtain_urls(self, query: str, modalities: List[str]) -> str:
        paraphrased_queries = self.paraphrase_agent.run_query(
            query=query, modalities=modalities
        )
        result_dict = self.orchestrator.run(
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

    def _infer(
        self, query: str, chat_history: List[Tuple[str, str]]
    ) -> Generator[Tuple[List[Tuple[str, str]], str], None, None]:
        chat_history = chat_history + [(query, "")]
        yield chat_history, ""

        msg, query, modalities = self._obtain_modes(query=query)
        chat_history[-1] = (query, msg)
        yield chat_history, ""
        result_text = self._obtain_urls(query=query, modalities=modalities)
        chat_history[-1] = (
            chat_history[-1][0],
            chat_history[-1][1] + "\n\n" + result_text,
        )
        yield chat_history, ""

    def launch_app(self):
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
        demo.launch()
