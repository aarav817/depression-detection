"""Regression checks for the interfaces brought together by the branch merge."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluate import binary_metrics, evaluate_binary, select_threshold


PROJECT = Path(__file__).resolve().parents[1]


class MergedInterfaceTests(unittest.TestCase):
    def test_metric_apis_keep_documented_undefined_values(self):
        project = binary_metrics([0, 1], [0, 0])
        custom = evaluate_binary([0, 1], [0, 0])
        self.assertEqual(project["precision"], 0)
        self.assertIsNone(custom["precision"])
        self.assertEqual(project["n"], custom["n_participants"])
        self.assertEqual(project["confusion_matrix"], custom["confusion_matrix"])
        self.assertEqual(evaluate_binary([0, 1], [-2, 2], 0)["balanced_accuracy"], 1)

    def test_threshold_tie_policies_are_explicit(self):
        labels, scores = [0, 1], [0.1, 0.1]
        self.assertEqual(select_threshold(labels, scores), 0.5)
        self.assertEqual(select_threshold(labels, scores, tie_break="higher"), np.nextafter(0.1, np.inf))

    def test_custom_training_and_evaluation_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame({
                "participant_key": list("abcdefgh"), "label": [0, 1] * 4,
                "split": ["train"] * 4 + ["validation"] * 2 + ["test"] * 2,
                "AU01_r_mean": [0, 1, np.nan, 2, 0.1, 1.5, "unused", "unused"],
            })
            frame.to_csv(root / "table.csv", index=False)
            (root / "features.json").write_text('["AU01_r_mean"]')
            trained = subprocess.run([
                sys.executable, str(PROJECT / "src/train.py"), str(root / "table.csv"),
                "--features", str(root / "features.json"), "--output", str(root / "output"),
            ], check=True, capture_output=True, text=True)
            report = json.loads(trained.stdout)
            self.assertEqual(report["logistic_regression"]["n_participants"], 2)
            bundle = joblib.load(root / "output/baseline.joblib")
            self.assertEqual(bundle["features"], ["AU01_r_mean"])
            subprocess.run([
                sys.executable, str(PROJECT / "src/evaluate.py"),
                str(root / "output/validation_predictions.csv"),
                "--threshold", str(bundle["threshold"]), "--output", str(root / "evaluation.json"),
            ], check=True, capture_output=True, text=True)
            evaluated = json.loads((root / "evaluation.json").read_text())
            self.assertEqual(evaluated["overall"], report["logistic_regression"])

    def test_evaluation_selects_one_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({
                "participant_key": ["a", "b", "a", "b"], "label": [0, 1, 0, 1],
                "score": [0.1, 0.9, 0, 0], "model": ["logistic", "logistic", "dummy", "dummy"],
            }).to_csv(root / "predictions.csv", index=False)
            subprocess.run([
                sys.executable, str(PROJECT / "src/evaluate.py"), str(root / "predictions.csv"),
                "--model", "logistic", "--threshold", "0.5", "--output", str(root / "report.json"),
            ], check=True, capture_output=True, text=True)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["overall"]["n_participants"], 2)
            self.assertEqual(report["overall"]["balanced_accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
