"""Participant-level binary evaluation; label 1 is the positive class."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def validate_inputs(labels, scores):
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or len(y) == 0 or len(y) != len(s):
        raise ValueError("Labels and scores must be nonempty, equal-length vectors.")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("Labels must be binary and scores must be finite.")
    return y.astype(int), s


def evaluate_binary(labels, scores, threshold=0.5):
    """Return JSON-safe metrics. Undefined rates are None, never silently zero."""
    y, s = validate_inputs(labels, scores)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite.")
    tn, fp, fn, tp = confusion_matrix(y, s >= threshold, labels=[0, 1]).ravel()
    ratio = lambda a, b: float(a / b) if b else None
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    both = len(np.unique(y)) == 2
    return {
        "n_participants": len(y), "n_positive": int(y.sum()),
        "threshold": float(threshold), "precision": ratio(tp, tp + fp),
        "recall": recall, "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2 if both else None,
        "average_precision": float(average_precision_score(y, s)) if y.sum() else None,
        "roc_auc": float(roc_auc_score(y, s)) if both else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def select_threshold(labels, scores):
    """Maximize validation balanced accuracy; ties prefer the higher threshold."""
    y, s = validate_inputs(labels, scores)
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires both classes in validation.")
    candidates = np.r_[np.unique(s), np.nextafter(s.max(), np.inf)]
    candidates = candidates[np.isfinite(candidates)]
    return float(max(candidates, key=lambda t: (
        evaluate_binary(y, s, t)["balanced_accuracy"], t)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="CSV: participant_key,label,score; optional subset/gender")
    parser.add_argument("--threshold", type=float, required=True, help="Previously frozen validation threshold")
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    if frame["participant_key"].isna().any() or frame["participant_key"].duplicated().any():
        raise ValueError("Expected exactly one row per participant.")
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
