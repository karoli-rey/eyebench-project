"""
Local runner script for DL models on SBSAT_STD.
Trains AhnRNN (gaze-only), Roberta (text-only), and RoberteyeWord (gaze+text)
across specified folds, then evaluates and aggregates results.

Uses subprocess to call the Hydra-based train.py and test_dl.py scripts,
with VRAM-optimized settings for Roberta-based models.

Usage:
    $env:WANDB_MODE="disabled"
    $env:CUDA_VISIBLE_DEVICES="0"
    python src/run/single_run/train_dl_local.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Configuration — Edit these to match your setup
# ---------------------------------------------------------------------------

# Which folds to train. Use [0,1,2,3] for full cross-validation,
# or [0,1] for a faster run (you already have fold 0 from previous runs).
FOLDS = [0, 1, 2, 3]

# Where Hydra should save outputs (checkpoints, configs).
OUTPUT_BASE = Path('outputs/sbsat_dl_local')


def get_canonical_run_name(model_name: str) -> str:
    """Match the ML result-folder convention so raw_to_processed can discover DL runs."""
    return (
        f'+data=SBSAT_STD,+model={model_name},+trainer=TrainerDL,'
        f'trainer.wandb_job_type={model_name}_SBSAT_STD'
    )

# Models to train with their Hydra config overrides.
# batch_size and accumulate_grad_batches are tuned to reduce VRAM.
MODELS = [
    {
        'name': 'AhnRNN',
        'overrides': [
            '+model=AhnRNN',
            # AhnRNN is lightweight — no VRAM optimization needed
        ],
    },
    {
        'name': 'Roberta',
        'overrides': [
            '+model=Roberta',
            'model.batch_size=2',
            'model.accumulate_grad_batches=8',
            'trainer.precision=SIXTEEN_MIXED',
        ],
    },
    {
        'name': 'RoberteyeWord',
        'overrides': [
            '+model=RoberteyeWord',
            'model.batch_size=2',
            'model.accumulate_grad_batches=8',
            'trainer.precision=SIXTEEN_MIXED',
        ],
    },
    {
        'name': 'RoberteyeLate',
        'overrides': [
            '+model=RoberteyeLateArgs',
            'model.batch_size=2',
            'model.accumulate_grad_batches=8',
            'trainer.precision=SIXTEEN_MIXED',
        ],
    }
]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model_name: str, overrides: list[str], fold: int) -> bool:
    """Train a single model on a single fold. Returns True on success."""

    output_dir = OUTPUT_BASE / get_canonical_run_name(model_name) / f'fold_index={fold}'

    # Skip if a checkpoint already exists (from a previous run)
    existing_ckpts = list(output_dir.glob('*.ckpt'))
    if existing_ckpts:
        logger.info(f'Checkpoint already exists for {model_name} fold {fold}, skipping training.')
        return True

    cmd = [
        sys.executable,
        'src/run/single_run/train.py',
        # Hydra output directory override
        f"hydra.run.dir='{output_dir}'",
        # Data config
        '+data=SBSAT_STD',
        f'data.fold_index={fold}',
        # Trainer config
        '+trainer=TrainerDL',
        'trainer.overwrite_data=True',
        'trainer.run_mode=TRAIN',
        'trainer.wandb_job_type=train',
        # Model-specific overrides
        *overrides,
    ]

    logger.info(f'Training {model_name} fold {fold}...')
    logger.info(f'  Output: {output_dir}')
    logger.info(f'  Command: {" ".join(cmd)}')

    result = subprocess.run(
        cmd,
        env={**os.environ, 'WANDB_MODE': 'disabled'},
        cwd=str(Path.cwd()),
    )

    if result.returncode != 0:
        logger.error(f'FAILED: {model_name} fold {fold} (exit code {result.returncode})')
        return False

    logger.info(f'SUCCESS: {model_name} fold {fold}')
    return True


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_model(model_name: str) -> bool:
    """Run test_dl.py on a trained model to generate prediction CSVs."""

    eval_path = OUTPUT_BASE / get_canonical_run_name(model_name)

    # Check that at least one fold directory exists with a checkpoint
    fold_dirs = sorted(eval_path.glob('fold_index=*'))
    if not fold_dirs:
        logger.warning(f'No fold directories found for {model_name}, skipping test.')
        return False

    has_ckpt = any(list(d.glob('*.ckpt')) for d in fold_dirs)
    if not has_ckpt:
        logger.warning(f'No checkpoints found for {model_name}, skipping test.')
        return False

    cmd = [
        sys.executable,
        'src/run/single_run/test_dl.py',
        f"eval_path='{eval_path}'",
    ]

    logger.info(f'Testing {model_name}...')
    logger.info(f'  eval_path: {eval_path}')

    result = subprocess.run(
        cmd,
        env={**os.environ, 'WANDB_MODE': 'disabled'},
        cwd=str(Path.cwd()),
    )

    if result.returncode != 0:
        logger.error(f'FAILED test for {model_name} (exit code {result.returncode})')
        return False

    logger.info(f'SUCCESS test for {model_name}')
    return True


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results() -> bool:
    """Run raw_to_processed_results.py to compute RMSE, MAE, R²."""
    cmd = [sys.executable, 'src/run/multi_run/raw_to_processed_results.py']

    logger.info('Aggregating results...')
    result = subprocess.run(cmd, cwd=str(Path.cwd()))

    if result.returncode != 0:
        logger.error(f'Aggregation failed (exit code {result.returncode})')
        return False

    logger.info('Aggregation complete. Check results/eyebench_benchmark_results/')
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info('=' * 60)
    logger.info('DL Model Training for SBSAT_STD')
    logger.info(f'Folds: {FOLDS}')
    logger.info(f'Models: {[m["name"] for m in MODELS]}')
    logger.info(f'Output: {OUTPUT_BASE}')
    logger.info('=' * 60)

    # Verify CUDA is available
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        logger.warning('CUDA_VISIBLE_DEVICES not set. Set it to e.g. "0" for GPU training.')
        logger.warning('AhnRNN can run on CPU; Roberta/RoberteyeWord need a GPU.')

    # --- Phase 1: Train ---
    results = {}
    for model in MODELS:
        model_name = model['name']
        results[model_name] = []

        logger.info(f'\n=== {model_name} ===')
        for fold in FOLDS:
            success = train_model(model_name, model['overrides'], fold)
            results[model_name].append((fold, success))

    # --- Report training results ---
    logger.info('\n' + '=' * 60)
    logger.info('Training Summary:')
    for model_name, fold_results in results.items():
        successes = sum(1 for _, s in fold_results if s)
        total = len(fold_results)
        status = '✓' if successes == total else '⚠'
        logger.info(f'  {status} {model_name}: {successes}/{total} folds')
    logger.info('=' * 60)

    # --- Phase 2: Test ---
    logger.info('\n=== Testing Models ===')
    for model in MODELS:
        test_model(model['name'])

    # --- Phase 3: Aggregate ---
    logger.info('\n=== Aggregating Results ===')
    aggregate_results()

    logger.info('\nAll done! Check results/eyebench_benchmark_results/ for the final tables.')


if __name__ == '__main__':
    main()
