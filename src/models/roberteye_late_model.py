from __future__ import annotations

from contextlib import nullcontext

import torch
from loguru import logger
from torch import nn
from transformers import AutoModel

from src.configs.constants import TaskTypes
from src.configs.data import DataArgs
from src.configs.models.dl.CustomRoberteye import RoberteyeLateArgs
from src.configs.trainers import TrainerDL
from src.models.base_model import BaseModel, register_model


@register_model
class RoberteyeLate(BaseModel):
    def __init__(
        self,
        model_args: RoberteyeLateArgs,
        trainer_args: TrainerDL,
        data_args: DataArgs,
    ):
        super().__init__(model_args, trainer_args, data_args)
        self.model_args = model_args

        self.text_backbone = AutoModel.from_pretrained(model_args.backbone)
        if model_args.freeze:
            for param in self.text_backbone.parameters():
                param.requires_grad = False
        else:
            logger.info('Text backbone is not frozen and will be fine-tuned.')

        text_dim = self.text_backbone.config.hidden_size
        eyes_dim: int = model_args.eyes_dim
        dropout: float = model_args.eye_projection_dropout

        self.eye_projection = nn.Sequential(
            nn.Linear(eyes_dim, text_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim // 2, text_dim),
        )

        fused_dim = text_dim * 2
        self.regression_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, fused_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim // 4, self.num_classes),
        )

        self.save_hyperparameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        eyes: torch.Tensor,
        eye_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        backbone_ctx = torch.no_grad() if self.model_args.freeze else nullcontext()
        with backbone_ctx:
            text_out = self.text_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        text_cls = text_out.last_hidden_state[:, 0, :]
        projected_eyes = self.eye_projection(eyes)

        if eye_mask is not None:
            mask = eye_mask.unsqueeze(-1).float()
            gaze_pooled = (projected_eyes * mask).sum(dim=1) / mask.sum(dim=1).clamp(
                min=1
            )
        else:
            gaze_pooled = projected_eyes.mean(dim=1)

        fused = torch.cat([text_cls, gaze_pooled], dim=1)
        return self.regression_head(fused)

    def shared_step(
        self, batch: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_data = self.unpack_batch(batch)
        labels = batch_data.labels

        assert batch_data.p_input_ids is not None, 'p_input_ids must be present'
        input_ids = batch_data.p_input_ids
        attention_mask = input_ids.ne(1).long()

        eyes = batch_data.eyes
        eye_mask = eyes.abs().sum(dim=-1).gt(0)

        logits = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            eyes=eyes,
            eye_mask=eye_mask,
        )

        if self.task == TaskTypes.REGRESSION:
            labels = labels.squeeze().float()
            logits = logits.squeeze()

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        loss = self.loss(logits, labels)
        return labels, loss, logits.squeeze()