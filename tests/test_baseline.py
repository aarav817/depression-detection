"""Verify metrics, leakage prevention, and participant alignment."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.dataset import FEATURE_COLUMNS
from src.evaluate import binary_metrics, select_threshold
from src.train import load_data, make_pipeline


class MetricsTests(unittest.TestCase):
    def test_hand_calculated_confusion_matrix(self):
        result = binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        for metric in ["accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy"]:
            self.assertEqual(result[metric], 0.5)
        self.assertAlmostEqual(result["roc_auc"], 0.75)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)

    def test_no_positive_predictions(self):
        result = binary_metrics([0, 0, 1], [0, 0, 0])
        self.assertEqual(result["precision"], 0)
        self.assertEqual(result["recall"], 0)
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(result["average_precision"], 1 / 3)

    def test_single_class_auc_is_undefined(self):
        result = binary_metrics([0, 0], [0.1, 0.8])
        for metric in ["roc_auc", "recall", "balanced_accuracy", "average_precision"]:
            self.assertIsNone(result[metric])

    def test_threshold_finds_perfect_boundary(self):
        threshold = select_threshold([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(threshold, 0.3)
        self.assertEqual(binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], threshold)["balanced_accuracy"], 1)

    def test_threshold_tie_prefers_default(self):
        self.assertEqual(select_threshold([0, 1], [0, 0]), 0.5)

    def test_bad_scores_are_rejected(self):
        for scores in [[np.nan, 0], [0.2, 1.1], [0.2]]:
            with self.assertRaises(ValueError):
                binary_metrics([0, 1], scores)


class TrainingTests(unittest.TestCase):
    def test_preprocessing_stays_fitted_to_training_data(self):
        train = pd.DataFrame({"value": [0.0, 2.0, np.nan, 4.0]})
        pipeline = make_pipeline(LogisticRegression())
        pipeline.fit(train, [0, 0, 1, 1])
        np.testing.assert_allclose(pipeline.named_steps["imputer"].statistics_, [2.0])
        np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_, [2.0])
        pipeline.predict_proba(pd.DataFrame({"value": [100000.0, np.nan]}))
        np.testing.assert_allclose(pipeline.named_steps["imputer"].statistics_, [2.0])
        np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_, [2.0])
        self.assertEqual(pipeline.named_steps["scaler"].n_samples_seen_, 4)

    def test_key_alignment_and_test_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keys = list("abcdef")
            pd.DataFrame({"participant_key": keys, "label": [0, 1, 0, 1, 0, 1],
                          "subset": ["HDS"] * 6, "gender": ["f"] * 6}).to_csv(root / "manifest.csv", index=False)
            splits = pd.DataFrame({"participant_key": keys, "split": ["train"] * 2 + ["validation"] * 2 + ["test"] * 2})
            splits.to_csv(root / "splits.csv", index=False)
            features = pd.DataFrame({"participant_key": keys, **{name: range(6) for name in FEATURE_COLUMNS}})
            features.iloc[::-1].to_csv(root / "features.csv", index=False)
            train, validation, test_count = load_data(root)
            self.assertEqual(train.participant_key.tolist(), ["a", "b"])
            self.assertEqual(validation.participant_key.tolist(), ["c", "d"])
            self.assertEqual(validation[FEATURE_COLUMNS[0]].tolist(), [2, 3])
            self.assertEqual(test_count, 2)
            # Alter held-out test features: training and validation inputs must be identical.
            features.loc[features.participant_key.isin(["e", "f"]), FEATURE_COLUMNS] = 999999
            features.to_csv(root / "features.csv", index=False)
            new_train, new_validation, _ = load_data(root)
            pd.testing.assert_frame_equal(train, new_train)
            pd.testing.assert_frame_equal(validation, new_validation)
            pd.concat([splits, splits.iloc[[0]]]).to_csv(root / "splits.csv", index=False)
            with self.assertRaises(ValueError):
                load_data(root)


if __name__ == "__main__":
    unittest.main()
