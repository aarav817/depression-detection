# Facial behavior research prototype

Predict AVDD-TCMI dataset labels from existing facial measurements. These scores are research outputs and do not establish a clinical diagnosis or a calibrated probability of depression.

## Setup (macOS or Ubuntu)

Use Python 3.10 or newer. On Ubuntu, install `python3-venv` and `python3-pip` with your system package manager if unavailable. On macOS, use a Python installation with venv support.

Run from the repository directory (`depression-detection`):

```sh
python3 -m venv .venv-research
source .venv-research/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Optional GRU dependency; CPU is sufficient for the synthetic checks:
python -m pip install -r requirements-neural.txt
python -m unittest discover -s tests -v
```

If using the existing environment one directory above this repository, activate `../.venv-research/bin/activate` instead. GPU installations require a PyTorch build compatible with your hardware; CUDA is not available on macOS.

## Layout

```text
src/download_data.py   Download archive and extract visual/label CSVs
src/inspect_data.py    Check files, create manifest, inspect example columns
src/split_data.py      Fixed participant splits stratified by subset and label
src/dataset.py         68 AU summary features and per-participant quality reports
src/evaluate.py        Metrics and frozen-threshold evaluation CLI
src/train.py           Dummy and logistic-regression training CLI
src/model.py           Variable-length GRU architecture (training loop pending)
tests/                 Synthetic metric, training, and GRU checks
reports/research_outline.md
requirements.txt       Core dependencies
requirements-neural.txt Optional PyTorch dependencies
```

## Data and feature preparation

```sh
python src/download_data.py --list-only
python src/download_data.py
python src/inspect_data.py
python src/split_data.py
python src/dataset.py
```

The downloader obtains the full ZIP because CSVs are archived; allow sufficient disk space. It records the dataset revision when downloading remotely. Verify dataset licensing and access terms before use or redistribution. Keep participant data and derived features out of the public repository.

The project workflow uses three separate tables in `data/processed/`: `manifest.csv` for labels and metadata, `splits.csv` for assignments, and `features.csv` for the 68 AU summaries. They join one-to-one using `participant_key`. `feature_quality.csv` records extraction issues. Keep the same saved splits for every model comparison.

The optional custom-table training interface accepts a combined CSV with exactly one row per participant:

```csv
participant_key,label,split,subset,AU01_r_mean,AU01_r_std
HDS_0,0,train,HDS,0.2,0.1
HDS_1,1,validation,HDS,0.8,0.3
```

This example documents the custom schema only: actual training needs both classes in training and validation. Use `train`, `validation`, or `test` for split values. Use only verified facial features, excluding metadata and tracking fields. For a custom table, store an explicit JSON feature list, for example `["AU01_r_mean", "AU01_r_std"]`, as `data/processed/custom_features.json`. Do not train on the manifest directly.

## Baseline and evaluation

Once the feature table exists:

```sh
python src/train.py
```

The baseline imputes training medians, standardizes using training statistics, and fits balanced logistic regression. It selects C from 0.1, 1, and 10 using validation average precision, then selects the threshold using validation balanced accuracy. C ties favor smaller C. In the project workflow, threshold ties favor proximity to 0.5, then higher thresholds. Test participants are excluded from fitting, selection, and prediction. Validation results are development estimates because they guide model and threshold selection.

`models/baseline/` contains `logistic_regression.joblib` and `dummy.joblib`, each storing its pipeline, ordered `feature_columns`, and frozen threshold. `results/baseline/` contains candidate comparisons, validation predictions for both models, aggregate and subgroup metrics, and run metadata with input hashes and dependency versions. Existing committed results are a historical baseline; generated results, models, and participant data are ignored by Git for future additions.

For a separately prepared combined table, the custom interface remains available:

```sh
python src/train.py data/processed/custom_table.csv --features data/processed/custom_features.json --output results/custom_baseline
```

This interface saves `baseline.joblib` with the keys `pipeline`, `features`, `threshold`, and `C`, plus the dummy pipeline, assignments, and validation reports. It retains the custom workflow's higher-threshold tie rule. Its default output is `results/custom_baseline` so it does not overwrite project results.

To evaluate the project's already-saved validation predictions, use the frozen threshold from `results/baseline/run_metadata.json`. The two-model CSV requires `--model` to select one model:

```sh
# Use the exact selected_threshold from the saved run metadata:
python src/evaluate.py results/baseline/validation_predictions.csv --model logistic_regression --threshold 0.4748287311261094 --output results/validation_evaluation.json
```

Only after all model decisions are frozen, the following example evaluates the reserved test group using the project artifact:

```python
import joblib
import pandas as pd

artifact = joblib.load("models/baseline/logistic_regression.joblib")  # Only load trusted artifacts.
frame = pd.read_csv("data/processed/features.csv").merge(
    pd.read_csv("data/processed/manifest.csv"), on="participant_key", validate="one_to_one"
).merge(pd.read_csv("data/processed/splits.csv"), on="participant_key", validate="one_to_one")
test = frame.loc[frame["split"] == "test"].copy()
test["score"] = artifact["pipeline"].predict_proba(test[artifact["feature_columns"]])[:, 1]
test[["participant_key", "label", "subset", "gender", "score"]].to_csv("results/test_predictions.csv", index=False)
print(artifact["threshold"])
```

```sh
# Replace 0.5 with the printed frozen threshold:
python src/evaluate.py results/test_predictions.csv --threshold 0.5
```

The evaluator reports precision, recall, specificity, balanced accuracy, average precision, ROC-AUC, and confusion matrix `[[TN, FP], [FN, TP]]`. Optional subset and gender columns produce group metrics. Scores at or above threshold are positive. `evaluate_binary()` and the CLI use JSON `null` for undefined rates and participant counts named `n_participants`/`n_positive`. The project report's `binary_metrics()` preserves its original zero-precision/F1 convention for zero denominators, with counts named `n`/`positive_count`; AUC and other rates requiring absent true classes remain `null`. Both APIs use shared metric calculations. Bootstrap intervals and plots remain future work.

## GRU

`FacialGRU(input_size)` accepts finite float tensors shaped `[batch, time, features]` and integer lengths shaped `[batch]`, returning one logit per recording. Padding is ignored via packed sequences; batches can have unsorted lengths. Defaults: hidden size 64, one layer, dropout 0.2. Use `BCEWithLogitsLoss`, training-only preprocessing, and training-derived class weights. Convert logits with sigmoid for scoring. Sequence preparation, participant-balanced training, early stopping, checkpointing, and participant score aggregation remain to be implemented before a real GRU experiment.

## Troubleshooting

- Missing imports: activate the intended environment and install with `python -m pip`.
- Missing visual files: run the downloader and check `data/raw/AVDD-TCMI-dataset` paths before inspection.
- Invalid training schema: confirm one row per participant, both classes in training/validation, and the explicit feature list.
- All-missing training feature: inspect extraction or remove that feature before fitting; never fill using validation/test statistics.
- GRU tests skipped: install `requirements-neural.txt`; the baseline does not require PyTorch.

See [the report outline](reports/research_outline.md) for the research question and limitations.
