"""Train baseline models and report validation results; reserve the test set.

Run from the project root: python src/train.py
"""

import argparse
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from .dataset import FEATURE_COLUMNS
    from .evaluate import binary_metrics, evaluate_binary, select_threshold
else:
    from dataset import FEATURE_COLUMNS
    from evaluate import binary_metrics, evaluate_binary, select_threshold


PROJECT = Path(__file__).resolve().parents[1]
SEED = 42
C_VALUES = (0.1, 1.0, 10.0)


def make_pipeline(classifier):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])


def load_data(processed):
    tables = {name: pd.read_csv(processed / f"{name}.csv") for name in ["manifest", "splits", "features"]}
    for name, table in tables.items():
        if table.participant_key.isna().any() or not table.participant_key.is_unique:
            raise ValueError(f"{name} must contain unique, nonmissing participant keys.")
        if set(table.participant_key) != set(tables["manifest"].participant_key):
            raise ValueError(f"{name} does not cover the manifest exactly. Review exclusions before training.")
    if set(tables["features"].columns) != {"participant_key", *FEATURE_COLUMNS}:
        raise ValueError("Feature table must contain exactly the 68 AU features and participant_key.")
    if set(tables["splits"].columns) != {"participant_key", "split"}:
        raise ValueError("Split table must contain participant_key and split only.")
    if set(tables["splits"]["split"]) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation, and test assignments.")
    if not tables["manifest"].label.isin([0, 1]).all():
        raise ValueError("Manifest labels must be 0 or 1.")
    joined = tables["manifest"][["participant_key", "label", "subset", "gender"]].merge(
        tables["splits"], on="participant_key", validate="one_to_one"
    ).merge(tables["features"], on="participant_key", validate="one_to_one")
    joined = joined.sort_values("participant_key").reset_index(drop=True)
    train = joined.loc[joined.split == "train"].copy()
    validation = joined.loc[joined.split == "validation"].copy()
    for name, group in [("train", train), ("validation", validation)]:
        if set(group.label) != {0, 1}:
            raise ValueError(f"{name} must contain both labels.")
        values = group[FEATURE_COLUMNS].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"Infinite features in {name}; review feature extraction.")
    # Only these two groups reach model fitting or prediction in this script.
    return train, validation, int((joined.split == "test").sum())


def fit_baselines(train, validation, feature_columns=FEATURE_COLUMNS, threshold_tie_break="closest_to_half", verbose=True):
    x_train, y_train = train[feature_columns], train.label
    x_validation, y_validation = validation[feature_columns], validation.label
    dummy = make_pipeline(DummyClassifier(strategy="most_frequent", random_state=SEED))
    dummy.fit(x_train, y_train)
    candidates, best_model, best_ap, best_c = [], None, -1.0, None
    for c_value in C_VALUES:
        model = make_pipeline(LogisticRegression(
            C=c_value, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=SEED,
        ))
        # Treat failed convergence as an error rather than saving an unreliable fit.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x_train, y_train)
        scores = model.predict_proba(x_validation)[:, 1]
        ap = float(average_precision_score(y_validation, scores))
        candidates.append({"C": c_value, **binary_metrics(y_validation, scores, 0.5)})
        if verbose:
            print(f"C={c_value:g}: validation average precision={ap:.4f}", flush=True)
        if ap > best_ap:  # Ties keep the first (smallest C) model.
            best_model, best_ap, best_c = model, ap, c_value

    scores = best_model.predict_proba(x_validation)[:, 1]
    threshold = select_threshold(y_validation, scores, tie_break=threshold_tie_break)
    return {"dummy": dummy, "logistic_regression": best_model}, candidates, best_c, threshold


def train_baselines(frame, features, output):
    """Custom-table API: one participant per row and an explicit feature list."""
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
    train = frame.loc[frame.split == "train"].copy()
    validation = frame.loc[frame.split == "validation"].copy()
    if train.label.nunique() != 2 or validation.label.nunique() != 2:
        raise ValueError("Train and validation must each contain both classes.")
    for group in (train, validation):
        for feature in features:
            group[feature] = pd.to_numeric(group[feature], errors="raise").replace([np.inf, -np.inf], np.nan)
    if train[features].isna().all().any():
        raise ValueError("A feature has no observed training values.")
    models, candidates, c_value, threshold = fit_baselines(
        train, validation, features, threshold_tie_break="higher", verbose=False
    )
    pipeline = models["logistic_regression"]
    scores = pipeline.predict_proba(validation[features])[:, 1]
    dummy_scores = models["dummy"].predict_proba(validation[features])[:, 1]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": features, "threshold": threshold, "C": c_value}, output / "baseline.joblib")
    joblib.dump(models["dummy"], output / "dummy.joblib")
    frame[["participant_key", "split"]].to_csv(output / "splits.csv", index=False)
    predictions = validation[[col for col in ("participant_key", "label", "subset", "gender") if col in validation]].copy()
    predictions["score"] = scores
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    report = {
        "selected_C": c_value, "selection_metric": "validation average precision",
        "threshold_tie_break": "higher", "test_evaluated": False,
        "candidates": [{"C": item["C"], "average_precision": item["average_precision"]} for item in candidates],
        "dummy": evaluate_binary(validation.label, dummy_scores),
        "logistic_regression": evaluate_binary(validation.label, scores, threshold),
    }
    (output / "validation_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def run_project_baseline():
    processed = PROJECT / "data" / "processed"
    train, validation, test_count = load_data(processed)
    print(f"Training: {len(train)}; validation: {len(validation)}; test reserved: {test_count}")
    models, candidates, best_c, threshold = fit_baselines(train, validation)
    results_dir = PROJECT / "results" / "baseline"
    model_dir = PROJECT / "models" / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics, prediction_tables, subgroup_rows = {}, [], []
    for name, model in models.items():
        cutoff = threshold if name == "logistic_regression" else 0.5
        scores = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
        metrics[name] = binary_metrics(validation.label, scores, cutoff)
        predictions = validation[["participant_key", "label", "subset", "gender"]].copy()
        predictions["model"] = name
        predictions["split"] = "validation"
        predictions["score"] = scores
        predictions["threshold"] = cutoff
        predictions["prediction"] = (scores >= cutoff).astype(int)
        prediction_tables.append(predictions)
        for column in ["subset", "gender"]:
            for group_name, group in predictions.groupby(column, dropna=False):
                subgroup_rows.append({
                    "model": name, "group_by": column, "group": str(group_name),
                    **binary_metrics(group.label, group.score, cutoff),
                })
        bundle = {"pipeline": model, "threshold": cutoff, "feature_columns": FEATURE_COLUMNS,
                  "positive_label": 1, "selected_C": best_c if name == "logistic_regression" else None}
        model_path = model_dir / f"{name}.joblib"
        joblib.dump(bundle, model_path)
        restored = joblib.load(model_path)
        np.testing.assert_allclose(restored["pipeline"].predict_proba(validation[FEATURE_COLUMNS])[:, 1], scores)

    default_scores = models["logistic_regression"].predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    metrics["logistic_regression_threshold_0.5"] = binary_metrics(validation.label, default_scores, 0.5)
    report = {
        "evaluation_split": "validation", "test_evaluated": False,
        "selection_note": "C and threshold were chosen on this validation set; these are development results, not final test estimates.",
        "confusion_matrix_order": "rows=true [0,1], columns=predicted [0,1]",
        "metrics": metrics, "subgroups": subgroup_rows,
    }
    (results_dir / "validation_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(candidates).drop(columns="confusion_matrix").to_csv(results_dir / "candidate_metrics.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(results_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(subgroup_rows).drop(columns="confusion_matrix").to_csv(results_dir / "subgroup_metrics.csv", index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "training_participants": len(train), "validation_participants": len(validation),
        "reserved_test_participants": test_count, "feature_columns": FEATURE_COLUMNS,
        "C_candidates": C_VALUES, "selected_C": best_c,
        "model_selection": "Highest validation average precision; ties prefer smaller C",
        "threshold_selection": "Highest validation balanced accuracy; ties prefer closest to 0.5, then higher threshold",
        "selected_threshold": threshold, "preprocessing_fit_split": "train",
        "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__, "joblib": joblib.__version__},
        "input_sha256": {name: hashlib.sha256((processed / name).read_bytes()).hexdigest()
                         for name in ["manifest.csv", "splits.csv", "features.csv"]},
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nSelected C={best_c:g}, validation threshold={threshold:.6f}")
    for name in models:
        result = metrics[name]
        print(f"{name}: balanced accuracy={result['balanced_accuracy']:.4f}, "
              f"AP={result['average_precision']:.4f}, recall={result['recall']:.4f}, "
              f"specificity={result['specificity']:.4f}")
    print(f"\nSaved models: {model_dir}\nSaved validation results: {results_dir}")
    print("The test set has not been evaluated.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", nargs="?", type=Path, help="Optional combined participant/label/split/feature CSV")
    parser.add_argument("--features", type=Path, help="JSON feature list, required for a custom table")
    parser.add_argument("--output", type=Path, help="Custom-table output folder (default: results/custom_baseline)")
    args = parser.parse_args()
    if args.table is None:
        if args.features is not None or args.output is not None:
            parser.error("--features and --output require a custom table path")
        run_project_baseline()
    else:
        if args.features is None:
            parser.error("A custom table requires --features")
        report = train_baselines(
            pd.read_csv(args.table), json.loads(args.features.read_text()),
            args.output or PROJECT / "results" / "custom_baseline",
        )
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
