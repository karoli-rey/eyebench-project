#### Gaze-Enhanced Prediction of Subjective Text Difficult

##### Project goal

This project investigates whether eye-tracking signals improve the prediction of subjective text difficulty. We use the SB-SAT dataset within the EyeBench framework and focus on the \`SBSAT\_STD\` regression task, where the target is the reader’s perceived difficulty rating for a text.

##### The main research question is:

*Does gaze information provide additional predictive value over text-only representations for subjective text difficulty prediction?*

#### Dataset and task

We use the SB-SAT dataset because it contains reading data aligned with eye-tracking measurements and includes subjective difficulty ratings. The task is formulated as a regression problem.
The evaluation follows the predefined EyeBench splits, including:
\- unseen reader
\- unseen text
\- unseen reader and text
The main metrics are:
\- RMSE
\- MAE
\- R²

# Timeline notes

## First step

Downloaded and preprocessed SBSAT via SBSAT\_STD commands
("bash src/data/preprocessing/get\_data.sh SBSAT")
see logs/preprocessing.log for details

## second step

Ran model checker for validating the cache, don't know if this is necessary to report

in the logs/failed\_runs.log and logs/completed\_runs.log it can be seen that some models required extra computing resources, which a 16GB RAM cpu only instance cannot handle

Then ran the command to remove debug runs

## third step

creating sweep configs

"bash run\_commands/utils/sweep\_wrapper.sh --data\_tasks SBSAT\_STD --folds 0,1,2,3 --wandb\_project SBSAT\_STD\_20251104 --regression true"

tried to train baseline logistic regression + random forest

## next steps?

### Deep Learning Model Selection Rationale

To address the core research question (*"Does gaze information provide additional predictive value over text-only representations?"*), we establish a three-way comparison using specific Deep Learning architectures:

1. **AhnRNN (Gaze-Only DL)**
   * **Why**: Represents the gaze-only baseline. It tests how much subjective text difficulty can be decoded purely from the physical mechanics of a reader's eye movements, independent of any language semantic understanding.
   * **What**: Models the temporal sequence of individual fixations, incorporating features like:
     * `CURRENT_FIX_DURATION` (duration of fixation)
     * `CURRENT_FIX_PUPIL` (pupil size/dilation)
     * `CURRENT_FIX_X` & `CURRENT_FIX_Y` (spatial coordinates)
   * **How**: Uses a lightweight LSTM layer (RNN) to process the sequence of fixations and pass the final hidden state to a regression layer. It runs very quickly even on CPUs.

2. **RoBERTa (Text-Only DL)**
   * **Why**: Represents the text-only baseline. It establishes the upper limit of difficulty prediction using only the static semantic complexity of the text, without any dynamic reader-response gaze data.
   * **What**: Utilizes a pretrained `roberta-large` language model.
   * **How**: Takes text strings, feeds them through the Transformer blocks, and runs a regression head over the representation of the `[CLS]` token.

3. **RoBERTEye-Word (Multimodal Gaze + Text)**
   * **Why**: The primary multimodal model. This tests if combining eye tracking features with language embeddings provides a statistically significant performance boost over the text-only and gaze-only baselines.
   * **What**: Combines word-level eye-movement features aligned with the corresponding RoBERTa word embeddings.
   * **How**: Projects the word-level gaze features into the Transformer's hidden space, concatenates/fuses them with the token embeddings, and passes the combined representation through the transformer blocks to predict difficulty.

Comparing these three models allows us to isolate the predictive value of text-only features, gaze-only features, and their fusion.

## baseline regression training

ran /single run on baseline dummy regression

#### results are logged in /results/eyebench_benchmark_results after running "python src/run/multi_run/raw_to_processed_results.py"

logs:
Saved rmse results to /files/eyebench-project/results/eyebench_benchmark_results/rmse.csv
Saved mae results to /files/eyebench-project/results/eyebench_benchmark_results/mae.csv
Saved r2 results to /files/eyebench-project/results/eyebench_benchmark_results/r2.csv
Saved results statistics for SBSAT_STD to /files/eyebench-project/results/eyebench_benchmark_results/stats_SBSAT_STD.csv

Need to double check which models are applicable for us:
Neural Models

- **AhnCNN** – CNN over fixation sequences (coordinates, durations, pupil size)  <- considered as gaze only
- **AhnRNN** – RNN variant of AhnCNN <- also considered if better than AhnCNN?
- **BEyeLSTM** – LSTM combining sequential fixations and global gaze statistics  
- **PLM-AS** – LSTM processing fixation-ordered word embeddings  
- **PLM-AS-RM** – RNN integrating fixation-ordered embeddings with reading measures  
- **RoBERTEye-W** – Transformer integrating word embeddings and word-level gaze features  
- **RoBERTEye-F** – Fixation-level variant of RoBERTEye-W  
- **MAG-Eye** – Multimodal Adaptation Gate injecting gaze into transformer layers  
- **PostFusion-Eye** – Cross-attention fusion of RoBERTa embeddings and CNN fixation features  

Traditional ML Models

- **Logistic / Linear Regression**  <-- ran this
- **Support Vector Machine (SVM / SVR)**  
- **Random Forest (Classifier / Regressor)**  

Baselines

- **Random** and **Majority Class** (classification)  
- **Mean** and **Median** (regression)  
- **Reading Speed**  
- **Text-Only RoBERTa** (no gaze input) <- need to look into this (already trained? or what)

Metrics

- **Regression:** RMSE, MAE, R²  
- **Aggregate:** Average Normalized Score and Mean Rank across all task–dataset pairs.

## further baseline model training (regression)

### Classical ML Baseline Selection Rationale

We focus on two classical machine learning baselines for comparison:
1. **DummyRegressor (Floor Baseline)**
   * **Why**: Establishes the absolute non-learning baseline (mean prediction floor). It proves that our actual models are learning meaningful patterns and is critical for contextualizing the negative $R^2$ scores typical of cross-validation splits.
2. **Support Vector Regressor (SVM / SVR)**
   * **Why**: Serves as our primary traditional ML baseline. It uses hand-crafted, aggregated gaze statistics (such as skip rate, mean fixation duration, and mean saccade velocity) computed from the reading trials. It shows what a robust, shallow machine learning model can achieve with aggregated gaze data before moving to deep temporal models like AhnRNN.

To keep our classical baselines focused and avoid redundant evaluations, we train only these two models. All models are run across 4 folds (0, 1, 2, 3).

The script is in src/run/single_run/test_ml_local.py

command: python src/run/single_run/test_ml_local.py

evaluation command: python src/run/multi_run/raw_to_processed_results.py

2026-05-27 13:02:07.826 | INFO     | __main__:save_metric_to_csv:530 - Saved rmse results to /files/eyebench-project/results/eyebench_benchmark_results/rmse.csv
2026-05-27 13:02:07.888 | INFO     | __main__:save_metric_to_csv:530 - Saved mae results to /files/eyebench-project/results/eyebench_benchmark_results/mae.csv
2026-05-27 13:02:07.941 | INFO     | __main__:save_metric_to_csv:530 - Saved r2 results to /files/eyebench-project/results/eyebench_benchmark_results/r2.csv
2026-05-27 13:02:07.949 | INFO     | __main__:save_metric_to_csv:539 - Saved results statistics for SBSAT_STD to /files/eyebench-project/results/eyebench_benchmark_results/stats_SBSAT_STD.csv

## Deep learning model training
ran this command: CUDA_VISIBLE_DEVICES=0 python src/run/single_run/train_dl_local.py


<br>
***

#### Current experimental status

So far, the work mainly follows the standard EyeBench workflow and scripts provided for the \`SBSAT\_STD\` task. This includes data preprocessing, model checking, sweep generation, training, and evaluation using the existing EyeBench infrastructure and implemented models.

##### Models already run

The current logs show that several existing EyeBench models were run on the \`SBSAT\_STD\` regression task.

Completed runs
\*\*Deep learning models\*\*
\- AhnCNN
\- AhnRNN
\- BEyeLSTM
\- PLMAS
\- PLMASf
\- MAG
\- PostFusion
\- Roberta
\- RoberteyeFixation
\- RoberteyeWord

\*\*Machine learning baselines\*\*
\- Support Vector Regressor
\- Random Forest Regressor
\- Linear Regression
\- Linear Meziere
\- Dummy Regressor

\#\#\# Failed or incomplete runs
Some larger deep learning models failed on later folds, especially folds 1–3:
\- MAG
\- PostFusion
\- Roberta
\- RoberteyeFixation
\- RoberteyeWord

These failures are likely related to resource or memory limitations rather than confirmed model implementation errors.

--> cont.
Failed or incomplete runs caused by RAM/VRAM leaks, however all ran without issues on CPU only with AT LEAST 28GB of RAM.
ML models run without issues on CPU only instance with 16GB RAM 
For DL models, we need to init an instance with Tesla T4 GPU that has 16GB of VRAM, pytorch requires minimum (not equal or less) 4GB VRAM