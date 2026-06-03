from __future__ import annotations

from contextlib import nullcontext

import torch
from loguru import logger
from torch import nn
from transformers import AutoModel

from src.configs.constants import TaskTypes
from src.configs.data import DataArgs
from src.configs.models.dl.CustomRoberteye import RoberteyeResidualGatedArgs
from src.configs.trainers import TrainerDL
from src.models.base_model import BaseModel, register_model


@register_model
class RoberteyeResidualGated(BaseModel):
    """
    Text-anchored residual gated fusion model.

    Instead of directly concatenating text and gaze, this model predicts:

        final = text_prediction + gate * gaze_residual

    The gate is learned per sample. If gaze is not useful, the model can keep
    the gate close to zero and behave like a text-only model.
    """

    def __init__(
        self,
        model_args: RoberteyeResidualGatedArgs,
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
            logger.info("Text backbone is not frozen and will be fine-tuned.")

        text_dim = self.text_backbone.config.hidden_size
        eyes_dim: int = model_args.eyes_dim

        eye_dropout = model_args.eye_projection_dropout
        attention_dropout = model_args.attention_dropout
        fusion_dropout = model_args.fusion_dropout
        gate_hidden_dim = model_args.gate_hidden_dim
        residual_hidden_dim = model_args.residual_hidden_dim

        self.eye_projection = nn.Sequential(
            nn.Linear(eyes_dim, text_dim // 2),
            nn.ReLU(),
            nn.Dropout(eye_dropout),
            nn.Linear(text_dim // 2, text_dim),
            nn.LayerNorm(text_dim),
        )

        # Text-conditioned attention over gaze tokens.
        # This replaces simple mean pooling.
        self.eye_attention = nn.Sequential(
            nn.Linear(text_dim * 2, text_dim // 2),
            nn.ReLU(),
            nn.Dropout(attention_dropout),
            nn.Linear(text_dim // 2, 1),
        )

        # Text-only prediction anchor.
        self.text_head = nn.Sequential(
            nn.Dropout(fusion_dropout),
            nn.Linear(text_dim, text_dim // 4),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(text_dim // 4, self.num_classes),
        )

        fusion_dim = text_dim * 4

        # Residual correction from gaze/text interaction.
        self.residual_head = nn.Sequential(
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_dim, residual_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(residual_hidden_dim, self.num_classes),
        )

        # Gate decides how much residual gaze correction to apply.
        self.gate = nn.Sequential(
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Initialize final gate bias so model begins close to text-only.
        final_gate_layer = self.gate[-2]
        if isinstance(final_gate_layer, nn.Linear):
            nn.init.constant_(final_gate_layer.bias, model_args.gate_init_bias)

        self.last_gate: torch.Tensor | None = None
        self.last_text_prediction: torch.Tensor | None = None
        self.last_residual_prediction: torch.Tensor | None = None

        if getattr(model_args, "use_huber_loss", False):
            beta = getattr(model_args, "huber_beta", 0.5)
            self.loss = nn.SmoothL1Loss(beta=beta)
            logger.info(f"Using SmoothL1Loss / Huber loss with beta={beta}")

        self.save_hyperparameters()


    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        MPS does not support float64 tensors. Some dataset features/labels may
        arrive as float64, so cast them to float32 before moving the batch.
        """

        def move_to_device(x):
            if isinstance(x, torch.Tensor):
                if x.dtype == torch.float64:
                    x = x.float()
                return x.to(device)

            if isinstance(x, dict):
                return {k: move_to_device(v) for k, v in x.items()}

            if isinstance(x, list):
                return [move_to_device(v) for v in x]

            if isinstance(x, tuple):
                return tuple(move_to_device(v) for v in x)

            return x

        return move_to_device(batch)
        

    def _attentive_gaze_pool(
        self,
        text_cls: torch.Tensor,
        projected_eyes: torch.Tensor,
        eye_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, n_eye_tokens, hidden_dim = projected_eyes.shape

        text_expanded = text_cls.unsqueeze(1).expand(
            batch_size,
            n_eye_tokens,
            hidden_dim,
        )

        attention_input = torch.cat([projected_eyes, text_expanded], dim=-1)
        attention_scores = self.eye_attention(attention_input).squeeze(-1)

        if eye_mask is not None:
            attention_scores = attention_scores.masked_fill(~eye_mask, -1e4)

        attention_weights = torch.softmax(attention_scores, dim=1)
        gaze_context = torch.sum(
            projected_eyes * attention_weights.unsqueeze(-1),
            dim=1,
        )

        return gaze_context

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

        gaze_context = self._attentive_gaze_pool(
            text_cls=text_cls,
            projected_eyes=projected_eyes,
            eye_mask=eye_mask,
        )

        interaction_features = torch.cat(
            [
                text_cls,
                gaze_context,
                text_cls * gaze_context,
                torch.abs(text_cls - gaze_context),
            ],
            dim=-1,
        )

        text_prediction = self.text_head(text_cls)
        residual_prediction = self.residual_head(interaction_features)
        gate = self.gate(interaction_features)

        self.last_gate = gate.detach()
        self.last_text_prediction = text_prediction.detach()
        self.last_residual_prediction = residual_prediction.detach()

        final_prediction = text_prediction + gate * residual_prediction
        return final_prediction

    def shared_step(
        self, batch: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_data = self.unpack_batch(batch)
        labels = batch_data.labels

        assert batch_data.p_input_ids is not None, "p_input_ids must be present"
        input_ids = batch_data.p_input_ids
        attention_mask = input_ids.ne(1).long()

        assert batch_data.eyes is not None, "eyes must be present"
        eyes = batch_data.eyes
        eye_mask = eyes.abs().sum(dim=-1).gt(0)

        logits = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            eyes=eyes,
            eye_mask=eye_mask,
        )

        if self.task == TaskTypes.REGRESSION:
            labels = labels.float().view(-1)
            logits = logits.float().view(-1)

        loss = self.loss(logits, labels)
        return labels, loss, logits  
        