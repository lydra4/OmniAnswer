import logging
from typing import Optional

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

    def _infer(self, query: str):
        self.content_moderator.moderate_query(query=query)
        modalities = self.modality_agent.run_query(query=query)
        paraphrased_queries = self.paraphrase_agent.run_query(
            query=query, modalities=modalities
        )
        result_dict = self.orchestrator.run(
            query=query, paraphrase_queries=paraphrased_queries
        )

    def launch_app(self):
        with gr.Blocks() as frontend:
            gr.Markdown(value=self.caption)
            chatbot = gr.Chatbot()
            query = gr.Textbox(show_label=True)

        frontend.launch()
