import logging
from typing import List, Optional, Tuple

import omegaconf

import gradio as gr
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
            msg = f"For your {query}, {mode_str} is the best mode to learn."
        elif n == 2:
            mode_str = f"{modalities[0]} and {modalities[1]}"
            msg = f"For your {query}, {mode_str} are the best mode to learn."
        else:
            mode_str = ", ".join(modalities[:-1]) + " and " + modalities[-1]
            msg = f"For your {query}, {mode_str} are the best mode to learn."

        return msg, query, modalities

    def _obtain_urls(self, query: str, modalities: List[str]) -> str:
        paraphrased_queries = self.paraphrase_agent.run_query(
            query=query, modalities=modalities
        )
        result_dict = self.orchestrator.run(
            query=query, paraphrase_queries=paraphrased_queries
        )
        result_text = "\n".join(
            [f"For {mode}: {url}" for mode, url in result_dict.items()]
        )
        return result_text

    def _infer(self, query: str) -> Tuple[str, str]:
        msg, query, modalities = self._obtain_modes(query=query)
        result_text = self._obtain_urls(query=query, modalities=modalities)
        return msg, result_text

    def launch_app(self):
        with gr.Blocks() as frontend:
            gr.Markdown(value=self.caption)
            query = gr.Textbox(
                show_label=True,
                label="Your Query",
                placeholder="A penny for your query.",
            )

            output_mode = gr.Textbox(
                label="Best Mode(s)",
                interactive=False,
            )

            output_url = gr.Textbox(
                label="URL(s) by mode",
                interactive=False,
            )

            ask_btn = gr.Button("Enter")
            ask_btn.click(
                fn=self._infer,
                inputs=[query],
                outputs=[output_mode, output_url],
            )

        frontend.launch()
