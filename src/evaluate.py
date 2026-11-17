"""Participant-level binary metrics and validation-only threshold selection."""

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def validate_predictions(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape or not len(labels):
        raise ValueError("Labels and scores must be nonempty, equal-length vectors.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Class-1 scores must be finite and between 0 and 1.")
    return labels.astype(int), scores


def binary_metrics(labels, scores, threshold=0.5):
    """Confusion matrix uses rows=true labels, columns=predictions, order [0, 1].

    Precision/F1 are zero when no positives are predicted. Metrics requiring a
    missing true class are None, so subgroup reports cannot imply a valid AUC.
    """
    labels, scores = validate_predictions(labels, scores)
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
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": sensitivity, "specificity": specificity,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "balanced_accuracy": (sensitivity + specificity) / 2 if both_classes else None,
        "average_precision": float(average_precision_score(labels, scores)) if labels.sum() else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def select_threshold(labels, scores):
    """Maximize validation balanced accuracy; ties prefer proximity to 0.5."""
    labels, scores = validate_predictions(labels, scores)
    if len(np.unique(labels)) != 2:
        raise ValueError("Threshold selection requires both labels in validation data.")
    # Include all distinct decisions, plus 0.5 and an all-negative option.
    candidates = np.unique(np.r_[0.0, 0.5, scores, np.nextafter(scores.max(), np.inf)])
    best_threshold, best_key = None, None
    for threshold in candidates:
        predictions = scores >= threshold
        sensitivity = predictions[labels == 1].mean()
        specificity = (~predictions[labels == 0]).mean()
        key = ((sensitivity + specificity) / 2, -abs(float(threshold) - 0.5), float(threshold))
        if best_key is None or key > best_key:
            best_threshold, best_key = float(threshold), key
    return best_threshold
