import json
import logging
import os
from typing import Dict

import torch
from omegaconf import DictConfig
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.text.bert import BERTScore


class EvaluationPipeline:
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.result_dict = self._load_results_dict(path=self.cfg.output_path)

        if "text" in self.result_dict.keys():
            self.text_metric = BERTScore(
                model_name_or_path=self.cfg.text_model_name,
                device=self.device,
            )

        if "image" in self.result_dict.keys():
            self.image_metrics = CLIPScore(
                model_name_or_path=self.cfg.image_model_name, device=self.device
            )

        if "video" in self.result_dict.keys():
            pass

    def _load_results_dict(self, path: str) -> Dict[str, str]:
        result_dict_path = os.path.join(path, "evaluation_dict.json")
        if os.path.exists(result_dict_path):
            try:
                with open(result_dict_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error occured: {e}.")
