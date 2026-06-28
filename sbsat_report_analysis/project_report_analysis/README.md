# SBSAT_STD Project Report Analysis

This folder contains the analysis notebook for the SBSAT_STD project.

The notebook loads the final experiment results, shows the main tables and plots, and explains the results in markdown.

## Notebook

`SBSAT_STD_results_analysis.ipynb`

## Project task

The task is to predict subjective text difficulty (`SBSAT_STD`) using text and eye-tracking data.

This is a regression task. The main evaluation metrics are:

* RMSE
* MAE
* R²

Classification metrics such as accuracy, F1 score, and ROC curves are not used.

## Main model

The main custom model is called:

`RoberteyeResidualGated`

The model combines a text-based prediction with a gaze-based residual correction:

`final prediction = text prediction + gate × gaze residual`

The gate controls how much the gaze information changes the text-based prediction.

## Source files used by the notebook

The notebook reads results from the main project folders:

* `../results/eyebench_benchmark_results/`
* `../outputs/rgated_eval/`
* `../outputs/rgated_huber_eval/`

These folders contain the benchmark tables and the trial-level prediction files.

## Main result

The residual-gated model improves over the existing late-fusion multimodal baseline, `RoberteyeLate`.

It does not clearly outperform the text-only `Roberta` baseline.

This suggests that gaze features can help, but the benefit depends on how they are combined with text features.
