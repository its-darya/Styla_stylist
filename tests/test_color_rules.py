import unittest

from ml.compatibility.rules import color_clash, hue_distance, pattern_clash
from ml.vision.color import rgb_to_hsv_spec


class TestHueDistance(unittest.TestCase):
    def test_same_hue_zero(self):
        self.assertEqual(hue_distance(0.0, 0.0), 0.0)

    def test_wraparound(self):
        self.assertAlmostEqual(hue_distance(10.0, 350.0), 20.0)


class TestColorClash(unittest.TestCase):
    def test_red_green_clashes(self):
        red = rgb_to_hsv_spec((255, 0, 0))
        green = rgb_to_hsv_spec((0, 255, 0))
        self.assertTrue(color_clash(red, green))

    def test_complementary_allowed(self):
        red = rgb_to_hsv_spec((255, 0, 0))
        cyan = rgb_to_hsv_spec((0, 255, 255))
        self.assertFalse(color_clash(red, cyan))

    def test_analogous_allowed(self):
        red = rgb_to_hsv_spec((255, 0, 0))
        orange = rgb_to_hsv_spec((255, 128, 0))
        self.assertFalse(color_clash(red, orange))

    def test_neutral_never_clashes(self):
        gray = rgb_to_hsv_spec((128, 128, 128))
        green = rgb_to_hsv_spec((0, 255, 0))
        self.assertFalse(color_clash(gray, green))


class TestPatternClash(unittest.TestCase):
    def test_two_patterns_clash(self):
        self.assertTrue(pattern_clash(["check", "striped"]))

    def test_solid_plus_pattern_ok(self):
        self.assertFalse(pattern_clash(["solid", "check"]))

    def test_all_solid_ok(self):
        self.assertFalse(pattern_clash(["solid", "solid"]))
