"""Train participant-level baselines, selecting C and threshold on validation only."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .evaluate import evaluate_binary, select_threshold
except ImportError:
    from evaluate import evaluate_binary, select_threshold


def train_baselines(frame, features, output):
    """Input: one row per participant, explicit feature allowlist, preassigned splits."""
    required = {"participant_key", "label", "split", *features}
    if not features or len(features) != len(set(features)) or not required.issubset(frame.columns):
        raise ValueError("Supply unique feature names and all required table columns.")
    forbidden = {"participant_key", "participant_id", "label", "split", "subset", "gender",
                 "timestamp", "frame", "confidence", "success", "visual_path", "n_frames"}
    if forbidden.intersection(features):
        raise ValueError("Metadata cannot be model features.")
    if frame.participant_key.isna().any() or frame.participant_key.duplicated().any():
        raise ValueError("Expected one row per participant across all splits.")
    if not frame.label.isin([0, 1]).all() or not frame.split.isin(["train", "validation", "test"]).all():
        raise ValueError("Expected binary labels and train/validation/test split values.")
    train = frame.loc[frame.split == "train"]
    val = frame.loc[frame.split == "validation"]
    if train.label.nunique() != 2 or val.label.nunique() != 2:
        raise ValueError("Train and validation must each contain both classes.")
    x_train = train[features].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    x_val = val[features].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    if x_train.isna().all().any():
        raise ValueError("A feature has no observed training values.")
    dummy = DummyClassifier(strategy="most_frequent").fit(x_train, train.label)
    dummy_scores = dummy.predict_proba(x_val)[:, list(dummy.classes_).index(1)]
    candidates = []
    for c in (0.1, 1.0, 10.0):
        pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=42))
        pipeline.fit(x_train, train.label)
        scores = pipeline.predict_proba(x_val)[:, 1]
        candidates.append((evaluate_binary(val.label, scores)["average_precision"], c, pipeline, scores))
    _, c, pipeline, scores = max(candidates, key=lambda item: item[0])
    threshold = select_threshold(val.label, scores)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": features, "threshold": threshold, "C": c}, output / "baseline.joblib")
    joblib.dump(dummy, output / "dummy.joblib")
    frame[["participant_key", "split"]].to_csv(output / "splits.csv", index=False)
    predictions = val[[col for col in ("participant_key", "label", "subset", "gender") if col in val]].copy()
    predictions["score"] = scores
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    report = {"selected_C": c, "selection_metric": "validation average precision",
              "candidates": [{"C": item[1], "average_precision": item[0]} for item in candidates],
              "dummy": evaluate_binary(val.label, dummy_scores),
              "logistic_regression": evaluate_binary(val.label, scores, threshold)}
    (output / "validation_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--features", type=Path, required=True, help="JSON list of feature column names")
    parser.add_argument("--output", type=Path, default=Path("results/baseline"))
    args = parser.parse_args()
    print(json.dumps(train_baselines(pd.read_csv(args.table), json.loads(args.features.read_text()), args.output), indent=2))


if __name__ == "__main__":
    main()
