"""
Local runner script for the DummyRegressor baseline on SBSAT_STD.
Trains DummyRegressor across all 4 folds without WandB.
Outputs trial_level_test_results.csv for each fold under results/raw/.

Usage:
    $env:WANDB_MODE="disabled"
    python src/run/single_run/test_ml_local.py
"""

from __future__ import annotations

import lightning_fabric as lf
import torch
from loguru import logger

from src.configs.data import SBSAT_STD
from src.configs.models.ml.DummyClassifier import DummyRegressorMLArgs
from src.configs.models.ml.SVM import SupportVectorRegressorMLArgs
from src.configs.trainers import TrainerML
from src.run.single_run.test_ml import process_single_run


def main():
    # Ensure reproducibility
    lf.seed_everything(42, workers=True, verbose=False)
    torch.set_float32_matmul_precision('high')

    # DummyRegressor only — the no-signal baseline (predicts training mean).
    # Gaze-only is handled by AhnRNN in train_dl_local.py.
    model_classes = [
        DummyRegressorMLArgs,
        SupportVectorRegressorMLArgs,
    ]

    for model_class in model_classes:
        model_name = model_class.__name__
        logger.info(f'=== Starting {model_name} ===')

        for fold in range(4):
            logger.info(f'--- Fold {fold} / {model_name} ---')

            data_args = SBSAT_STD(fold_index=fold)
            trainer_args = TrainerML(
                run_mode='train',
                wandb_job_type='train',
                overwrite_data=True,
            )
            model_args = model_class()

            process_single_run(
                data_args=data_args,
                trainer_args=trainer_args,
                model_args=model_args,
                fold_index=fold,
            )

        logger.info(f'=== Finished {model_name} across all 4 folds ===')

    logger.info('All models complete. Run raw_to_processed_results.py to aggregate.')


if __name__ == '__main__':
    main()
