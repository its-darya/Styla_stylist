# tests/test_generate.py
import random
import unittest

from ml.compatibility.generate import (
    Outfit,
    WardrobeItem,
    generate_outfit,
    outfit_is_valid,
)


def item(cat, iid, true=None):
    return WardrobeItem(item_id=iid, category=cat, true_category=true or cat)


class TestGenerate(unittest.TestCase):
    def test_top_bottom_valid(self):
        outfit = generate_outfit([item("top", "t1"), item("bottom", "b1")], rng=random.Random(0))
        self.assertTrue(outfit_is_valid(outfit))

    def test_dress_valid(self):
        outfit = generate_outfit([item("dress", "d1")], rng=random.Random(0))
        self.assertTrue(outfit_is_valid(outfit))

    def test_misclassified_dress_as_top_is_invalid(self):
        items = [
            WardrobeItem(item_id="d1", category="top", true_category="dress"),
            item("bottom", "b1"),
        ]
        outfit = generate_outfit(items, rng=random.Random(0))
        self.assertFalse(outfit_is_valid(outfit))

    def test_dress_plus_bottom_invalid(self):
        outfit = Outfit(items=[
            WardrobeItem(item_id="d1", category="dress", true_category="dress"),
            WardrobeItem(item_id="b1", category="bottom", true_category="bottom"),
        ])
        self.assertFalse(outfit_is_valid(outfit))

    def test_missing_core_invalid(self):
        outfit = Outfit(items=[item("shoes", "s1")])
        self.assertFalse(outfit_is_valid(outfit))
