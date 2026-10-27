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
src/dataset.py         Pending: participant splits and summary features
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
```

The downloader obtains the full ZIP because CSVs are archived; allow sufficient disk space. It records the dataset revision when downloading remotely. Verify dataset licensing and access terms before use or redistribution. Keep participant data and derived features out of the public repository.

The training interface expects a CSV with exactly one row per participant:

```csv
participant_key,label,split,subset,AU01_r_mean,AU01_r_std
HDS_0,0,train,HDS,0.2,0.1
HDS_1,1,validation,HDS,0.8,0.3
```

This example documents the schema only: actual training needs both classes in training and validation. Use `train`, `validation`, or `test` for split values. Create permanent approximately 70/15/15 participant splits stratified by subset and label; no participant may span splits. Aggregate multiple recordings per participant before baseline training. Use only verified facial features, excluding metadata and tracking fields. Store an explicit JSON feature list, for example `["AU01_r_mean", "AU01_r_std"]`, as `data/processed/features.json`.

Feature extraction and split creation in `src/dataset.py` remain to be implemented. Do not train on the manifest directly.

## Baseline and evaluation

Once the feature table exists:

```sh
python src/train.py data/processed/features.csv --features data/processed/features.json
```

The baseline imputes training medians, standardizes using training statistics, and fits balanced logistic regression. It selects C from 0.1, 1, and 10 using validation average precision, then selects the threshold using validation balanced accuracy. Ties favor smaller C and higher thresholds. Test feature values are not read during fitting or selection.

`results/baseline/` contains the fitted pipeline, ordered feature names, frozen threshold, selected C, dummy classifier, split assignments, validation predictions, and metrics. Keep the source feature table, dataset revision, dependency versions (`python -m pip freeze`), and feature-extraction settings alongside the experiment.

To evaluate a previously frozen model, use its saved feature order and threshold:

```python
import joblib
import pandas as pd

artifact = joblib.load("results/baseline/baseline.joblib")  # Only load trusted artifacts.
frame = pd.read_csv("data/processed/features.csv")
test = frame.loc[frame["split"] == "test"].copy()
test["score"] = artifact["pipeline"].predict_proba(test[artifact["features"]])[:, 1]
test.to_csv("results/test_predictions.csv", index=False)
print(artifact["threshold"])
```

```sh
# Replace 0.5 with the printed frozen threshold:
python src/evaluate.py results/test_predictions.csv --threshold 0.5
```

The evaluator reports precision, recall, specificity, balanced accuracy, average precision, ROC-AUC, and confusion matrix `[[TN, FP], [FN, TP]]`. Optional subset and gender columns produce group metrics. Undefined metrics are JSON `null`; scores at or above threshold are positive. Evaluate the test set only after freezing decisions. Bootstrap intervals and plots remain future work.

## GRU

`FacialGRU(input_size)` accepts finite float tensors shaped `[batch, time, features]` and integer lengths shaped `[batch]`, returning one logit per recording. Padding is ignored via packed sequences; batches can have unsorted lengths. Defaults: hidden size 64, one layer, dropout 0.2. Use `BCEWithLogitsLoss`, training-only preprocessing, and training-derived class weights. Convert logits with sigmoid for scoring. Sequence preparation, participant-balanced training, early stopping, checkpointing, and participant score aggregation remain to be implemented before a real GRU experiment.

## Troubleshooting

- Missing imports: activate the intended environment and install with `python -m pip`.
- Missing visual files: run the downloader and check `data/raw/AVDD-TCMI-dataset` paths before inspection.
- Invalid training schema: confirm one row per participant, both classes in training/validation, and the explicit feature list.
- All-missing training feature: inspect extraction or remove that feature before fitting; never fill using validation/test statistics.
- GRU tests skipped: install `requirements-neural.txt`; the baseline does not require PyTorch.

See [the report outline](reports/research_outline.md) for the research question and limitations.
