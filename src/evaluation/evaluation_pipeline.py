import logging
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
        result_dict: Dict[str, str],
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.result_dict = result_dict

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
