# tests/test_evaluate.py
import unittest

from ml.evaluate import classification_report, threshold_sweep
from ml.vision.classify import ClassifiedItem

LABELS = ("top", "bottom", "dress", "outerwear", "shoes")


def scenario():
    return [
        ClassifiedItem("t1", "top", "top", 0.90),
        ClassifiedItem("b1", "bottom", "bottom", 0.85),
        ClassifiedItem("b2", "bottom", "bottom", 0.80),
        ClassifiedItem("d1", "dress", "dress", 0.90),
        ClassifiedItem("d2", "dress", "top", 0.40),
        ClassifiedItem("s1", "shoes", "shoes", 0.70),
        ClassifiedItem("o1", "outerwear", "outerwear", 0.70),
    ]


class TestClassificationReport(unittest.TestCase):
    def test_accuracy_and_macro_f1(self):
        report = classification_report(scenario(), LABELS)
        self.assertAlmostEqual(report["accuracy"], 6 / 7)
        self.assertGreater(report["macro_f1"], 0.0)


class TestThresholdSweep(unittest.TestCase):
    def test_higher_threshold_improves_validity(self):
        results = threshold_sweep(
            scenario(), LABELS, thresholds=[0.0, 0.5], n_outfits=200, seed=0
        )
        self.assertGreater(results[1]["valid_outfit_rate"], results[0]["valid_outfit_rate"])
        self.assertEqual(results[1]["kept"], 6)
        self.assertEqual(results[0]["kept"], 7)
