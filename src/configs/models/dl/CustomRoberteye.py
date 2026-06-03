"""CustomRoberteye.py
Configuration for the RoberteyeLate (late-fusion) model.

This model separates gaze and text modality processing:
  - Text tokens pass through a *frozen* roberta-large backbone.
  - Gaze features are projected by a small trainable MLP.
  - Both representations are concatenated and fed to a regression head.

Because the heavy Transformer weights are frozen, backpropagation only flows
through the lightweight projection and output layers, drastically reducing
VRAM requirements compared to the deep-fusion RoberteyeWord baseline.
"""

from dataclasses import dataclass

from src.configs.constants import BackboneNames, DLModelNames
from src.configs.models.base_model import DLModelArgs
from src.configs.utils import register_model_config


@register_model_config
@dataclass
class RoberteyeLateArgs(DLModelArgs):
    """
    Late-fusion RoBERTeye configuration.

    Inherits from DLModelArgs and sets sensible defaults for the late-fusion
    architecture that can run on a single GPU with limited VRAM (or even CPU).

    Attributes:
        base_model_name: Links to DLModelNames.ROBERTEYE_LATE so the
            ModelFactory resolves the correct class.
        backbone: Frozen RoBERTa backbone used to extract text embeddings.
        freeze: Whether to freeze backbone parameters (should normally be True
            for late fusion to keep VRAM low).
        eye_projection_dropout: Dropout applied in the gaze projection MLP.
        batch_size / accumulate_grad_batches: Tuned for low VRAM usage.
        prepend_eye_features_to_text: True so the data pipeline provides gaze
            features in the batch (they are *not* prepended in the model; only
            used for projection).
        n_tokens / eye_token_id / sep_token_id / is_training: Dynamically set
            by ``_configure_roberteye_model`` in ``train.py``.
    """

    base_model_name: DLModelNames = DLModelNames.ROBERTEYE_LATE

    prepend_eye_features_to_text: bool = True
    use_fixation_report: bool = False  # Uses word-level gaze, not fixation-level
    batch_size: int = 4
    accumulate_grad_batches: int = 16 // batch_size
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    freeze: bool = True  # Default True: only projection + head are trained
    eye_projection_dropout: float = 0.3
    max_epochs: int = 10
    early_stopping_patience: int = 3
    warmup_proportion: float = 0.1

    # Token bookkeeping — filled at runtime by train.py
    token_type_num: int = 2
    vocab_size: int = -1  # specified in instantiate_config
    n_tokens: int = 0
    eye_token_id: int = 0
    sep_token_id: int = 0
    is_training: bool = False


@register_model_config
@dataclass
class RoberteyeResidualGatedArgs(RoberteyeLateArgs):
    """
    Text-anchored residual gated late-fusion model.

    Starts from the same setup as RoberteyeLate but predicts:
        final_prediction = text_prediction + gate * gaze_residual

    This allows the model to use gaze only when useful.
    """

    base_model_name: DLModelNames = DLModelNames.ROBERTEYE_RESIDUAL_GATED

    eye_projection_dropout: float = 0.2
    attention_dropout: float = 0.2
    fusion_dropout: float = 0.2
    gate_hidden_dim: int = 512
    residual_hidden_dim: int = 512

    # Negative bias means the model initially relies mostly on text.
    # sigmoid(-2.0) ≈ 0.12
    gate_init_bias: float = -2.0


@register_model_config
@dataclass
class RoberteyeResidualGatedSmallArgs(RoberteyeResidualGatedArgs):
    """
    Smaller and more conservative residual-gated fusion model.

    Uses the same RoberteyeResidualGated architecture, but reduces the
    capacity of the gate/residual networks and makes the initial gaze
    contribution smaller.
    """

    # Important: use the same registered model class.
    base_model_name: DLModelNames = DLModelNames.ROBERTEYE_RESIDUAL_GATED

    # More conservative gaze/fusion settings
    eye_projection_dropout: float = 0.3
    attention_dropout: float = 0.3
    fusion_dropout: float = 0.4

    gate_hidden_dim: int = 128
    residual_hidden_dim: int = 128

    # sigmoid(-3.0) ≈ 0.05, so gaze starts with ~5% influence.
    gate_init_bias: float = -3.0

@register_model_config
@dataclass
class RoberteyeResidualGatedMediumArgs(RoberteyeResidualGatedArgs):
    """
    Medium conservative residual-gated fusion model.

    This uses the same RoberteyeResidualGated architecture, but with
    intermediate capacity between the original and small versions.
    """

    # Use the same registered model class
    base_model_name: DLModelNames = DLModelNames.ROBERTEYE_RESIDUAL_GATED

    # Keep training settings here so we do not need long command-line overrides
    batch_size: int = 2
    accumulate_grad_batches: int = 8

    # Medium conservative fusion settings
    eye_projection_dropout: float = 0.25
    attention_dropout: float = 0.25
    fusion_dropout: float = 0.3

    gate_hidden_dim: int = 256
    residual_hidden_dim: int = 256

    # sigmoid(-2.5) ≈ 0.08, so gaze starts with ~8% influence
    gate_init_bias: float = -2.5


@register_model_config
@dataclass
class RoberteyeResidualGatedHuberArgs(RoberteyeResidualGatedArgs):
    """
    Original residual-gated model trained with SmoothL1/Huber loss.

    This keeps the best architecture:
        gate_hidden_dim = 512
        residual_hidden_dim = 512
        fusion_dropout = 0.2
        gate_init_bias = -2.0

    Only the loss function changes from MSE to SmoothL1Loss.
    """

    base_model_name: DLModelNames = DLModelNames.ROBERTEYE_RESIDUAL_GATED

    batch_size: int = 2
    accumulate_grad_batches: int = 8

    use_huber_loss: bool = True
    huber_beta: float = 0.5