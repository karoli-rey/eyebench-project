"""Backward-compatible surprisal extractor factory."""

from psycholing_metrics.surprisal.factory import create_surprisal_extractor


def get_surp_extractor(
    extractor_type,
    model_name: str,
    model_target_device: str = 'cpu',
    pythia_checkpoint: str | None = 'step143000',
    hf_access_token: str | None = None,
):
    return create_surprisal_extractor(
        extractor_type=extractor_type,
        model_name=model_name,
        model_target_device=model_target_device,
        pythia_checkpoint=pythia_checkpoint,
        hf_access_token=hf_access_token,
    )

