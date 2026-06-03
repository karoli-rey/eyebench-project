"""Backward-compatible wrapper for linguistic metrics extraction."""

from psycholing_metrics.metrics import get_metrics as _get_metrics


def get_metrics(*args, **kwargs):
    metrics_df = _get_metrics(*args, **kwargs)
    # Older EyeBench code expects `<model>_Surprisal` instead of `<model>_cat_Surprisal`.
    rename_map = {
        col: col.replace('_cat_Surprisal', '_Surprisal')
        for col in metrics_df.columns
        if col.endswith('_cat_Surprisal')
    }
    if rename_map:
        metrics_df = metrics_df.rename(columns=rename_map)
    return metrics_df

