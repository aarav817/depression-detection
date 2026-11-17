"""Participant-level binary metrics and validation-only threshold selection."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def validate_inputs(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape or not len(labels):
        raise ValueError("Labels and scores must be nonempty, equal-length vectors.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    return labels.astype(int), scores


def validate_predictions(labels, scores):
    labels, scores = validate_inputs(labels, scores)
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Class-1 scores must be between 0 and 1.")
    return labels, scores


def _metric_values(labels, scores, threshold, undefined_precision):
    """Shared calculations; callers validate their supported score formats."""
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite.")
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(labels, predictions, labels=[0, 1]).ravel())
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    both_classes = sensitivity is not None and specificity is not None
    return {
        "n": len(labels), "positive_count": int(labels.sum()),
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(labels),
        "precision": tp / (tp + fp) if tp + fp else undefined_precision,
        "recall": sensitivity, "specificity": specificity,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else undefined_precision,
        "balanced_accuracy": (sensitivity + specificity) / 2 if both_classes else None,
        "average_precision": float(average_precision_score(labels, scores)) if labels.sum() else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def binary_metrics(labels, scores, threshold=0.5):
    """Confusion matrix uses rows=true labels, columns=predictions, order [0, 1].

    Precision/F1 are zero when no positives are predicted. Metrics requiring a
    missing true class are None, so subgroup reports cannot imply a valid AUC.
    """
    labels, scores = validate_predictions(labels, scores)
    return _metric_values(labels, scores, threshold, undefined_precision=0.0)


def evaluate_binary(labels, scores, threshold=0.5):
    """Evaluation CLI/custom-table API: finite scores, undefined rates as None."""
    labels, scores = validate_inputs(labels, scores)
    metrics = _metric_values(labels, scores, threshold, undefined_precision=None)
    metrics["n_participants"] = metrics.pop("n")
    metrics["n_positive"] = metrics.pop("positive_count")
    return metrics


def select_threshold(labels, scores, tie_break="closest_to_half"):
    """Maximize validation balanced accuracy with an explicit tie rule.

    The project workflow retains its original proximity-to-0.5 tie rule. The
    custom-table workflow passes 'higher' to retain its previous behavior.
    """
    labels, scores = validate_inputs(labels, scores)
    if len(np.unique(labels)) != 2:
        raise ValueError("Threshold selection requires both labels in validation data.")
    if tie_break not in {"closest_to_half", "higher"}:
        raise ValueError("Unknown threshold tie rule.")
    # Include all distinct decisions, plus 0.5 and an all-negative option.
    candidates = np.unique(np.r_[0.0, 0.5, scores, np.nextafter(scores.max(), np.inf)])
    candidates = candidates[np.isfinite(candidates)]
    if tie_break == "higher":
        candidates = np.unique(np.r_[scores, np.nextafter(scores.max(), np.inf)])
        candidates = candidates[np.isfinite(candidates)]
    best_threshold, best_key = None, None
    for threshold in candidates:
        predictions = scores >= threshold
        sensitivity = predictions[labels == 1].mean()
        specificity = (~predictions[labels == 0]).mean()
        key = ((sensitivity + specificity) / 2,
               -abs(float(threshold) - 0.5) if tie_break == "closest_to_half" else 0.0,
               float(threshold))
        if best_key is None or key > best_key:
            best_threshold, best_key = float(threshold), key
    return best_threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="CSV: participant_key,label,score; optional subset/gender/model")
    parser.add_argument("--threshold", type=float, required=True, help="Previously frozen validation threshold")
    parser.add_argument("--model", help="Select one model when the CSV contains multiple models")
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    if not {"participant_key", "label", "score"}.issubset(frame.columns):
        raise ValueError("Predictions must contain participant_key, label, and score.")
    if args.model:
        if "model" not in frame:
            raise ValueError("--model requires a model column.")
        frame = frame.loc[frame.model == args.model]
    if frame["participant_key"].isna().any() or frame["participant_key"].duplicated().any():
        raise ValueError("Expected exactly one row per participant; use --model for multi-model CSVs.")
    report = {"overall": evaluate_binary(frame.label, frame.score, args.threshold)}
    for field in ("subset", "gender"):
        if field in frame:
            report[field] = {str(group): evaluate_binary(rows.label, rows.score, args.threshold)
                             for group, rows in frame.groupby(field, dropna=False)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
