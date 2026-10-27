"""Small hand-checkable cases for participant feature extraction."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.dataset import AU_COLUMNS, FEATURE_COLUMNS, summarize_visual


class FeatureExtractionTests(unittest.TestCase):
    def summarize(self, frame):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visual.csv"
            frame.to_csv(path, index=False)
            return summarize_visual(path)

    def frame(self, values, success):
        return pd.DataFrame({" success": success, **{f" {au}": values for au in AU_COLUMNS}})

    def test_statistics_and_failed_tracking_filter(self):
        features, quality = self.summarize(self.frame([1, 3, 999], [1, 1, 0]))
        self.assertEqual(len(features), 68)
        self.assertEqual(list(features), FEATURE_COLUMNS)
        self.assertEqual([features[f"AU01_r_{stat}"] for stat in ["mean", "std", "median", "iqr"]], [2, 1, 2, 1])
        self.assertEqual(quality["discarded_frames"], 1)
        self.assertEqual(quality["status"], "ok")

    def test_no_successful_frames_is_reported(self):
        features, quality = self.summarize(self.frame([1, 2], [0, 0]))
        self.assertIsNone(features)
        self.assertEqual(quality["status"], "excluded")
        self.assertIn("No successfully tracked", quality["reason"])

    def test_invalid_values_are_reported_and_not_imputed(self):
        frame = self.frame([1, 3], [1, 1])
        frame[" AU01_r"] = [float("inf"), "bad"]
        frame[" AU02_r"] = [-1, 3]
        features, quality = self.summarize(frame)
        self.assertTrue(pd.isna(features["AU01_r_mean"]))
        self.assertEqual(features["AU02_r_mean"], 3)
        self.assertEqual(quality["invalid_au_values"], 2)
        self.assertEqual(quality["out_of_range_au_values"], 1)
        self.assertEqual(quality["missing_features"], 4)
        self.assertEqual(quality["status"], "warning")

    def test_missing_column_is_reported(self):
        features, quality = self.summarize(self.frame([1], [1]).drop(columns=[" AU01_r"]))
        self.assertIsNone(features)
        self.assertIn("AU01_r", quality["reason"])

    def test_no_usable_values_is_reported(self):
        features, quality = self.summarize(self.frame([float("nan")], [1]))
        self.assertIsNone(features)
        self.assertIn("No usable AU", quality["reason"])


if __name__ == "__main__":
    unittest.main()
