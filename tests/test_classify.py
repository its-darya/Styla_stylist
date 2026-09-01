import unittest

from ml.vision.classify import ClassifiedItem, filter_by_confidence


def make(rows):
    return [ClassifiedItem(*row) for row in rows]


class TestFilterByConfidence(unittest.TestCase):
    def test_filters_below_threshold(self):
        items = make([
            ("a", "top", "top", 0.9),
            ("b", "dress", "top", 0.4),
        ])
        kept = filter_by_confidence(items, 0.5)
        self.assertEqual([i.item_id for i in kept], ["a"])

    def test_inclusive_at_threshold(self):
        items = make([("a", "top", "top", 0.5)])
        self.assertEqual(len(filter_by_confidence(items, 0.5)), 1)

    def test_empty(self):
        self.assertEqual(filter_by_confidence([], 0.5), [])
