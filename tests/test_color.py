import unittest

import numpy as np
from PIL import Image

from ml.vision.color import (
    dominant_color,
    extract_dominant_colors,
    rgb_to_hsv_spec,
)


def _solid_image(rgb, size=(32, 32), alpha=255):
    arr = np.zeros((size[0], size[1], 4), dtype=np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = rgb
    arr[..., 3] = alpha
    return Image.fromarray(arr, mode="RGBA")


class TestExtractDominantColors(unittest.TestCase):
    def test_solid_red_returns_red(self):
        clusters = extract_dominant_colors(_solid_image((255, 0, 0)), k=3)
        r, g, b = dominant_color(clusters)
        self.assertGreater(r, 200)
        self.assertLess(g, 60)
        self.assertLess(b, 60)

    def test_two_color_image_two_clusters(self):
        half = np.zeros((32, 32, 4), dtype=np.uint8)
        half[:16, :, :3] = (255, 0, 0)
        half[16:, :, :3] = (0, 0, 255)
        half[..., 3] = 255
        img = Image.fromarray(half, mode="RGBA")
        clusters = extract_dominant_colors(img, k=2, seed=0)
        self.assertEqual(len(clusters), 2)
        self.assertAlmostEqual(clusters[0].proportion, 0.5, delta=0.15)

    def test_ignores_transparent_pixels(self):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, :3] = (255, 255, 255)
        arr[:, :, 3] = 0
        arr[:, :16, :3] = (255, 0, 0)
        arr[:, :16, 3] = 255
        img = Image.fromarray(arr, mode="RGBA")
        clusters = extract_dominant_colors(img, k=1)
        r, g, b = dominant_color(clusters)
        self.assertGreater(r, 200)
        self.assertLess(g, 60)
        self.assertLess(b, 60)

    def test_fully_transparent_raises(self):
        with self.assertRaises(ValueError):
            extract_dominant_colors(_solid_image((255, 0, 0), alpha=0), k=3)


class TestRgbToHsv(unittest.TestCase):
    def test_red_hue_zero(self):
        spec = rgb_to_hsv_spec((255, 0, 0))
        self.assertAlmostEqual(spec.hue, 0.0, delta=1.0)
        self.assertAlmostEqual(spec.saturation, 1.0, delta=0.05)

    def test_green_hue_120(self):
        spec = rgb_to_hsv_spec((0, 255, 0))
        self.assertAlmostEqual(spec.hue, 120.0, delta=1.0)

    def test_gray_is_unsaturated(self):
        spec = rgb_to_hsv_spec((128, 128, 128))
        self.assertAlmostEqual(spec.saturation, 0.0, delta=0.05)
