from pathlib import Path
import ast
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RESULTS_DIR = Path("results/eyebench_benchmark_results")
PRED_PATHS = sorted(Path("outputs/rgated_eval").glob("fold_index=*/trial_level_test_results.csv"))

MODEL_NAME = "RoberteyeResidualGated"
DATA_NAME = "SBSAT_STD"

REGIME_DISPLAY = {
    "seen_subject_unseen_item": "Seen subject unseen item",
    "unseen_subject_seen_item": "Unseen subject seen item",
    "unseen_subject_unseen_item": "Unseen subject unseen item",
}

def clean_prediction(x):
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, (list, tuple, np.ndarray)):
                return float(parsed[0])
            return float(parsed)
        except Exception:
            return float(x)
    return float(x)

def metric_value(y_true, y_pred, metric_name):
    if metric_name == "rmse":
        return mean_squared_error(y_true, y_pred) ** 0.5
    if metric_name == "mae":
        return mean_absolute_error(y_true, y_pred)
    if metric_name == "r2":
        return r2_score(y_true, y_pred)
    raise ValueError(f"Unknown metric: {metric_name}")

def format_mean_sem(values):
    values = pd.Series(values).dropna()
    mean = values.mean()
    sem = values.sem()
    if pd.isna(sem):
        sem = 0.0
    return f"{mean:.2f} ± {sem:.2f}"

if not RESULTS_DIR.exists():
    raise SystemExit(f"Results folder not found: {RESULTS_DIR}")

if not PRED_PATHS:
    raise SystemExit("No trial_level_test_results.csv files found under outputs/rgated_eval.")

print("Found prediction files:")
for p in PRED_PATHS:
    print(" ", p)

if len(PRED_PATHS) != 4:
    print(f"\nWARNING: Expected 4 folds, but found {len(PRED_PATHS)} file(s).")

rows = []

for path in PRED_PATHS:
    df = pd.read_csv(path)

    fold = int(path.parent.name.split("=")[1])
    pred_col = "prediction_prob" if "prediction_prob" in df.columns else "prediction"
    label_col = "label" if "label" in df.columns else "labels"

    df[pred_col] = df[pred_col].apply(clean_prediction)
    df[label_col] = df[label_col].astype(float)

    for eval_type in sorted(df["eval_type"].unique()):
        eval_df = df[df["eval_type"] == eval_type]

        for regime_key in list(REGIME_DISPLAY.keys()) + ["all"]:
            if regime_key == "all":
                sub = eval_df
            else:
                sub = eval_df[eval_df["eval_regime"] == regime_key]

            if sub.empty:
                continue

            y_true = sub[label_col].to_numpy()
            y_pred = sub[pred_col].to_numpy()

            for metric_name in ["rmse", "mae", "r2"]:
                rows.append({
                    "Eval Type": eval_type,
                    "Data": DATA_NAME,
                    "Model": MODEL_NAME,
                    "eval_regime": regime_key,
                    "fold": fold,
                    "metric": metric_name,
                    "score": metric_value(y_true, y_pred, metric_name),
                })

fold_stats = pd.DataFrame(rows)

fold_stats_path = RESULTS_DIR / "stats_RoberteyeResidualGated_fold_level.csv"
fold_stats.to_csv(fold_stats_path, index=False)

for metric_name in ["rmse", "mae", "r2"]:
    old_path = RESULTS_DIR / f"{metric_name}.csv"

    if not old_path.exists():
        raise SystemExit(f"Could not find old benchmark file: {old_path}")

    old = pd.read_csv(old_path)
    new_rows = []

    for eval_type in ["val", "test"]:
        metric_df = fold_stats[
            (fold_stats["metric"] == metric_name) &
            (fold_stats["Eval Type"] == eval_type)
        ]

        row = {
            "Eval Type": eval_type,
            "Data": DATA_NAME,
            "Model": MODEL_NAME,
        }

        for regime_key, display_name in REGIME_DISPLAY.items():
            vals = metric_df[metric_df["eval_regime"] == regime_key]["score"]
            row[display_name] = format_mean_sem(vals)

        vals_all = metric_df[metric_df["eval_regime"] == "all"]["score"]
        row["All"] = format_mean_sem(vals_all)

        new_rows.append(row)

    new_df = pd.DataFrame(new_rows)

    # Remove old copy if script is rerun.
    combined = old[old["Model"] != MODEL_NAME].copy()
    combined = pd.concat([combined, new_df], ignore_index=True)

    out_path = RESULTS_DIR / f"{metric_name}_with_RoberteyeResidualGated.csv"
    combined.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(
        combined[
            (combined["Data"] == DATA_NAME) &
            (combined["Eval Type"] == "test")
        ].tail(10)
    )

print("\nSaved fold-level stats:")
print(fold_stats_path)
