import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import evaluate_binary, select_threshold
from src.train import train_baselines


class EvaluationTests(unittest.TestCase):
    def test_known_confusion(self):
        m = evaluate_binary([0, 0, 1, 1], [.1, .8, .4, .9])
        self.assertEqual(m['confusion_matrix'], [[1, 1], [1, 1]])
        for name in ('precision', 'recall', 'specificity', 'balanced_accuracy'):
            self.assertEqual(m[name], .5)
        self.assertAlmostEqual(m['roc_auc'], .75)
        self.assertAlmostEqual(m['average_precision'], 5 / 6)

    def test_undefined_and_invalid(self):
        m = evaluate_binary([0, 0], [.1, .2])
        self.assertIsNone(m['roc_auc'])
        self.assertIsNone(m['precision'])
        self.assertIsNone(m['average_precision'])
        for y, s in (([], []), ([2], [.5]), ([1], [np.nan])):
            with self.assertRaises(ValueError):
                evaluate_binary(y, s)

    def test_validation_threshold(self):
        self.assertEqual(select_threshold([0, 1], [.2, .4]), .4)

    def test_training_ignores_test_features(self):
        frame = pd.DataFrame({'participant_key': list('abcdefgh'),
                              'label': [0, 1, 0, 1, 0, 1, 0, 1],
                              'split': ['train'] * 4 + ['validation'] * 2 + ['test'] * 2,
                              'AU01_r_mean': [0., 1., np.nan, 2., .1, 1.5, 'unused', 'unused']})
        with tempfile.TemporaryDirectory() as directory:
            report = train_baselines(frame, ['AU01_r_mean'], directory)
            self.assertTrue((Path(directory) / 'baseline.joblib').is_file())
            self.assertEqual(report['logistic_regression']['n_participants'], 2)
        frame.loc[7, 'participant_key'] = 'a'
        with self.assertRaises(ValueError):
            train_baselines(frame, ['AU01_r_mean'], '.')


try:
    import torch
    from src.model import FacialGRU
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install requirements-neural.txt to test GRU')
class ModelTests(unittest.TestCase):
    def test_shape_padding_and_gradients(self):
        torch.manual_seed(42)
        model = FacialGRU(3).eval()
        x = torch.randn(3, 8, 3)
        lengths = torch.tensor([3, 8, 5])
        scores = model(x, lengths)
        self.assertEqual(tuple(scores.shape), (3,))
        modified = x.clone()
        modified[0, 3:] = 100
        modified[2, 5:] = -100
        torch.testing.assert_close(scores, model(modified, lengths))
        torch.nn.functional.binary_cross_entropy_with_logits(scores, torch.tensor([0., 1., 0.])).backward()
        self.assertIsNotNone(model.gru.weight_ih_l0.grad)
        with self.assertRaises(ValueError):
            model(x, [0, 8, 5])


if __name__ == '__main__':
    unittest.main()
