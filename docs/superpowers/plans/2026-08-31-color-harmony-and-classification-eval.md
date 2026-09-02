# Color Harmony + Classification-Impact Evaluation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add (1) K-means dominant-color extraction with a color-clash rule (alongside a minimal pattern-clash rule), and (2) a harness that measures how misclassified wardrobe items degrade outfit generation and tunes the confidence threshold.

**Architecture:** Three small, focused modules under `ml/` mirroring the README's planned layout — `ml/vision/color.py` (color extraction), `ml/vision/classify.py` (classification results + confidence filtering), `ml/compatibility/` (rules + generate stub) — wired together by `ml/evaluate.py`. Two CLI scripts reproduce the existing `scripts/test_background.py` pattern for hands-on runs.

**Tech Stack:** Python 3.14, numpy, Pillow, scikit-image (`rgb2lab`/`lab2rgb`), stdlib `unittest` + `colorsys` + `random`.

## Global Constraints

- Run Python with the project venv: `.venv/bin/python` (numpy/skimage live there).
- **No new third-party dependencies.** Use only numpy, Pillow, scikit-image, stdlib.
- Test runner: `.venv/bin/python -m unittest discover -s tests -t .` (from repo root).
- Category taxonomy: `("top", "bottom", "dress", "outerwear", "shoes")`.
- Code style: match existing — `from __future__` where helpful, `str | None` unions, `@dataclass`, type hints, no comments unless the logic is non-obvious.
- Commit style: `feat: ...` (matches repo history).

---

### Task 1: Dominant-color extraction (`ml/vision/color.py`)

**Files:**
- Create: `ml/vision/color.py`
- Modify: `ml/vision/__init__.py`
- Test: `tests/test_color.py`

**Interfaces:**
- Produces: `ColorCluster(rgb, proportion)`, `ColorSpec(hue, saturation, value)`, `extract_dominant_colors(image, k=3, *, min_alpha=128, seed=0, iters=10, max_pixels=20000) -> list[ColorCluster]`, `dominant_color(clusters) -> tuple[int,int,int]`, `rgb_to_hsv_spec(rgb) -> ColorSpec`. Consumed by Task 2 (rules) and Task 6 (CLI).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_color.py
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
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `.venv/bin/python -m unittest discover -s tests -t .`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml.vision.color'`.

- [ ] **Step 3: Write the implementation**

```python
# ml/vision/color.py
from dataclasses import dataclass

import colorsys

import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab


@dataclass
class ColorCluster:
    rgb: tuple[int, int, int]
    proportion: float


@dataclass
class ColorSpec:
    hue: float
    saturation: float
    value: float


def _kmeans(pixels_lab: np.ndarray, k: int, seed: int, iters: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = pixels_lab.shape[0]
    if k >= n:
        return pixels_lab.copy()

    centers = np.empty((k, 3), dtype=np.float32)
    centers[0] = pixels_lab[rng.integers(n)]
    for i in range(1, k):
        dists = ((pixels_lab[:, None, :] - centers[None, :i, :]) ** 2).sum(axis=2).min(axis=1)
        total = dists.sum()
        if total <= 0:
            centers[i] = pixels_lab[rng.integers(n)]
        else:
            centers[i] = pixels_lab[rng.choice(n, p=dists / total)]

    for _ in range(iters):
        dists = ((pixels_lab[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        for i in range(k):
            cluster = pixels_lab[labels == i]
            if cluster.shape[0] > 0:
                centers[i] = cluster.mean(axis=0)
    return centers


def extract_dominant_colors(
    image: Image.Image,
    k: int = 3,
    *,
    min_alpha: int = 128,
    seed: int = 0,
    iters: int = 10,
    max_pixels: int = 20000,
) -> list[ColorCluster]:
    if k < 1:
        raise ValueError("k must be >= 1")

    arr = np.asarray(image.convert("RGBA")).astype(np.float32)
    rgb = (arr[..., :3] / 255.0).reshape(-1, 3)
    alpha = arr[..., 3].reshape(-1)

    pixels_rgb = rgb[alpha >= min_alpha]
    if pixels_rgb.shape[0] == 0:
        raise ValueError("image has no opaque pixels")

    if pixels_rgb.shape[0] > max_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pixels_rgb.shape[0], max_pixels, replace=False)
        pixels_rgb = pixels_rgb[idx]

    k_eff = min(k, pixels_rgb.shape[0])
    pixels_lab = rgb2lab(pixels_rgb).astype(np.float32)
    centers_lab = _kmeans(pixels_lab, k_eff, seed, iters)
    centers_rgb = lab2rgb(centers_lab)

    dists = ((pixels_lab[:, None, :] - centers_lab[None, :, :]) ** 2).sum(axis=2)
    labels = dists.argmin(axis=1)
    counts = np.bincount(labels, minlength=k_eff).astype(float)
    proportions = counts / counts.sum()

    order = np.argsort(-proportions)
    return [
        ColorCluster(
            rgb=tuple((np.clip(centers_rgb[i], 0, 1) * 255).round().astype(int)),
            proportion=float(proportions[i]),
        )
        for i in order
    ]


def dominant_color(clusters: list[ColorCluster]) -> tuple[int, int, int]:
    if not clusters:
        raise ValueError("no clusters provided")
    return clusters[0].rgb


def rgb_to_hsv_spec(rgb: tuple[int, int, int]) -> ColorSpec:
    r, g, b = (c / 255.0 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return ColorSpec(hue=h * 360.0, saturation=s, value=v)
```

- [ ] **Step 4: Update `ml/vision/__init__.py`**

```python
# ml/vision/__init__.py
from ml.vision.background import BackgroundRemovalError, remove_background
from ml.vision.color import (
    ColorCluster,
    ColorSpec,
    dominant_color,
    extract_dominant_colors,
    rgb_to_hsv_spec,
)

__all__ = [
    "remove_background",
    "BackgroundRemovalError",
    "ColorCluster",
    "ColorSpec",
    "dominant_color",
    "extract_dominant_colors",
    "rgb_to_hsv_spec",
]
```

- [ ] **Step 5: Run tests, confirm pass**

Run: `.venv/bin/python -m unittest discover -s tests -t .`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add ml/vision/color.py ml/vision/__init__.py tests/test_color.py
git commit -m "feat: add K-means dominant color extraction"
```

---

### Task 2: Harmony rules (`ml/compatibility/rules.py`)

**Files:**
- Create: `ml/compatibility/__init__.py`
- Create: `ml/compatibility/rules.py`
- Test: `tests/test_color_rules.py`

**Interfaces:**
- Consumes: `ColorSpec`, `rgb_to_hsv_spec` from Task 1.
- Produces: `hue_distance(h1, h2) -> float`, `is_neutral(spec, sat_threshold=0.15) -> bool`, `color_clash(a, b, *, sat_threshold=0.15, low=45.0, high=135.0) -> bool`, `pattern_clash(patterns) -> bool`. Consumed by Task 6.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_color_rules.py
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
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `.venv/bin/python -m unittest discover -s tests -t .`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml.compatibility'`.

- [ ] **Step 3: Write the implementation**

```python
# ml/compatibility/rules.py
from ml.vision.color import ColorSpec

SOLID_PATTERN_ALIASES = {"solid", "plain", "none", "", "unpatterned"}

DEFAULT_NEUTRAL_SATURATION = 0.15
DEFAULT_CLASH_LOW = 45.0
DEFAULT_CLASH_HIGH = 135.0


def hue_distance(h1: float, h2: float) -> float:
    d = abs(h1 - h2) % 360.0
    return min(d, 360.0 - d)


def is_neutral(spec: ColorSpec, sat_threshold: float = DEFAULT_NEUTRAL_SATURATION) -> bool:
    return spec.saturation < sat_threshold


def color_clash(
    a: ColorSpec,
    b: ColorSpec,
    *,
    sat_threshold: float = DEFAULT_NEUTRAL_SATURATION,
    low: float = DEFAULT_CLASH_LOW,
    high: float = DEFAULT_CLASH_HIGH,
) -> bool:
    if is_neutral(a, sat_threshold) or is_neutral(b, sat_threshold):
        return False
    d = hue_distance(a.hue, b.hue)
    return low < d < high


def pattern_clash(patterns: list[str]) -> bool:
    strong = [p for p in patterns if p.strip().lower() not in SOLID_PATTERN_ALIASES]
    return len(strong) >= 2
```

```python
# ml/compatibility/__init__.py
from ml.compatibility.rules import color_clash, hue_distance, is_neutral, pattern_clash

__all__ = ["color_clash", "hue_distance", "is_neutral", "pattern_clash"]
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `.venv/bin/python -m unittest discover -s tests -t .`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/compatibility/__init__.py ml/compatibility/rules.py tests/test_color_rules.py
git commit -m "feat: add color-clash and pattern-clash rules"
```

---

### Task 3: Generate stub (`ml/compatibility/generate.py`)

**Files:**
- Create: `ml/compatibility/generate.py`
- Modify: `ml/compatibility/__init__.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Produces: `CATEGORIES`, `WardrobeItem(item_id, category, true_category=None, color=None, pattern=None)`, `Outfit(items)`, `generate_outfit(items, rng=random) -> Outfit`, `outfit_is_valid(outfit, *, use_true=True) -> bool`. Consumed by Task 5 and Task 6.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `.venv/bin/python -m unittest discover -s tests -t .`
Expected: FAIL — `ImportError` on `generate`.

- [ ] **Step 3: Write the implementation**

```python
# ml/compatibility/generate.py
import random
from dataclasses import dataclass

CATEGORIES = ("top", "bottom", "dress", "outerwear", "shoes")


@dataclass
class WardrobeItem:
    item_id: str
    category: str
    true_category: str | None = None
    color: tuple[int, int, int] | None = None
    pattern: str | None = None


@dataclass
class Outfit:
    items: list[WardrobeItem]


def generate_outfit(items: list[WardrobeItem], rng: random.Random = random) -> Outfit:
    tops = [i for i in items if i.category == "top"]
    bottoms = [i for i in items if i.category == "bottom"]
    dresses = [i for i in items if i.category == "dress"]
    shoes = [i for i in items if i.category == "shoes"]
    outer = [i for i in items if i.category == "outerwear"]

    chosen: list[WardrobeItem] = []
    if dresses and rng.random() >= 0.5:
        chosen.append(rng.choice(dresses))
    elif tops and bottoms:
        chosen.append(rng.choice(tops))
        chosen.append(rng.choice(bottoms))
    if shoes and rng.random() >= 0.5:
        chosen.append(rng.choice(shoes))
    if outer and rng.random() >= 0.5:
        chosen.append(rng.choice(outer))
    return Outfit(items=chosen)


def outfit_is_valid(outfit: Outfit, *, use_true: bool = True) -> bool:
    def cat(i: WardrobeItem) -> str:
        return i.true_category if (use_true and i.true_category) else i.category

    cats = [cat(i) for i in outfit.items]
    counts = {c: cats.count(c) for c in CATEGORIES}

    has_dress = counts["dress"] >= 1
    has_top = counts["top"] >= 1
    has_bottom = counts["bottom"] >= 1

    if has_dress:
        core_ok = counts["dress"] == 1 and not has_top and not has_bottom
    else:
        core_ok = has_top and has_bottom and counts["top"] == 1 and counts["bottom"] == 1

    return core_ok and counts["shoes"] <= 1 and counts["outerwear"] <= 1
```

- [ ] **Step 4: Update `ml/compatibility/__init__.py`**

```python
# ml/compatibility/__init__.py
from ml.compatibility.generate import (
    CATEGORIES,
    Outfit,
    WardrobeItem,
    generate_outfit,
    outfit_is_valid,
)
from ml.compatibility.rules import color_clash, hue_distance, is_neutral, pattern_clash

__all__ = [
    "CATEGORIES",
    "Outfit",
    "WardrobeItem",
    "generate_outfit",
    "outfit_is_valid",
    "color_clash",
    "hue_distance",
    "is_neutral",
    "pattern_clash",
]
```

- [ ] **Step 5: Run tests, confirm pass**

Run: `.venv/bin/python -m unittest discover -s tests -t .`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ml/compatibility/generate.py ml/compatibility/__init__.py tests/test_generate.py
git commit -m "feat: add rule-based outfit generator stub"
```

---

### Task 4: Classification results (`ml/vision/classify.py`)

**Files:**
- Create: `ml/vision/classify.py`
- Modify: `ml/vision/__init__.py`
- Test: `tests/test_classify.py`

**Interfaces:**
- Produces: `ClassifiedItem(item_id, true_category, predicted_category, confidence)`, `filter_by_confidence(items, threshold) -> list[ClassifiedItem]`. Consumed by Task 5/6.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_classify.py
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
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `.venv/bin/python -m unittest discover -s tests -t .`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml.vision.classify'`.

- [ ] **Step 3: Write the implementation**

```python
# ml/vision/classify.py
from dataclasses import dataclass


@dataclass
class ClassifiedItem:
    item_id: str
    true_category: str
    predicted_category: str
    confidence: float


def filter_by_confidence(
    items: list[ClassifiedItem],
    threshold: float,
) -> list[ClassifiedItem]:
    return [item for item in items if item.confidence >= threshold]
```

- [ ] **Step 4: Update `ml/vision/__init__.py`**

```python
# ml/vision/__init__.py
from ml.vision.background import BackgroundRemovalError, remove_background
from ml.vision.classify import ClassifiedItem, filter_by_confidence
from ml.vision.color import (
    ColorCluster,
    ColorSpec,
    dominant_color,
    extract_dominant_colors,
    rgb_to_hsv_spec,
)

__all__ = [
    "remove_background",
    "BackgroundRemovalError",
    "ClassifiedItem",
    "filter_by_confidence",
    "ColorCluster",
    "ColorSpec",
    "dominant_color",
    "extract_dominant_colors",
    "rgb_to_hsv_spec",
]
```

- [ ] **Step 5: Run tests, confirm pass**

Run: `.venv/bin/python -m unittest discover -s tests -t .`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ml/vision/classify.py ml/vision/__init__.py tests/test_classify.py
git commit -m "feat: add classification result types and confidence filter"
```

---

### Task 5: Evaluation harness (`ml/evaluate.py`)

**Files:**
- Create: `ml/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `ClassifiedItem`/`filter_by_confidence` (Task 4), `generate_outfit`/`outfit_is_valid`/`WardrobeItem` (Task 3).
- Produces: `classification_report(items, labels) -> dict`, `valid_outfit_rate(items, *, n_outfits=100, seed=0) -> float`, `threshold_sweep(items, labels, *, thresholds, n_outfits=100, seed=0) -> list[dict]`. Consumed by Task 6.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `.venv/bin/python -m unittest discover -s tests -t .`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml.evaluate'`.

- [ ] **Step 3: Write the implementation**

```python
# ml/evaluate.py
import random

from ml.compatibility.generate import WardrobeItem, generate_outfit, outfit_is_valid
from ml.vision.classify import ClassifiedItem, filter_by_confidence


def classification_report(
    items: list[ClassifiedItem],
    labels: tuple[str, ...],
) -> dict:
    y_true = [i.true_category for i in items]
    y_pred = [i.predicted_category for i in items]

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(items) if items else 0.0

    per_class = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(labels) if labels else 0.0
    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class}


def _to_wardrobe_items(items: list[ClassifiedItem]) -> list[WardrobeItem]:
    return [
        WardrobeItem(
            item_id=item.item_id,
            category=item.predicted_category,
            true_category=item.true_category,
        )
        for item in items
    ]


def valid_outfit_rate(
    items: list[ClassifiedItem],
    *,
    n_outfits: int = 100,
    seed: int = 0,
) -> float:
    wardrobe = _to_wardrobe_items(items)
    rng = random.Random(seed)
    valid = 0
    for _ in range(n_outfits):
        if outfit_is_valid(generate_outfit(wardrobe, rng=rng), use_true=True):
            valid += 1
    return valid / n_outfits if n_outfits else 0.0


def threshold_sweep(
    items: list[ClassifiedItem],
    labels: tuple[str, ...],
    *,
    thresholds: list[float],
    n_outfits: int = 100,
    seed: int = 0,
) -> list[dict]:
    results = []
    for threshold in thresholds:
        kept = filter_by_confidence(items, threshold)
        report = classification_report(kept, labels)
        results.append({
            "threshold": threshold,
            "kept": len(kept),
            "total": len(items),
            "accuracy": report["accuracy"],
            "macro_f1": report["macro_f1"],
            "valid_outfit_rate": valid_outfit_rate(kept, n_outfits=n_outfits, seed=seed),
        })
    return results
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `.venv/bin/python -m unittest discover -s tests -t .`
Expected: PASS (all suites).

- [ ] **Step 5: Commit**

```bash
git add ml/evaluate.py tests/test_evaluate.py
git commit -m "feat: add evaluation harness for classification impact on generation"
```

---

### Task 6: CLI scripts (`scripts/`)

**Files:**
- Create: `scripts/test_color.py`
- Create: `scripts/test_evaluate.py`

**Interfaces:**
- Consumes: `extract_dominant_colors`, `dominant_color`, `rgb_to_hsv_spec` (Task 1); `threshold_sweep`, `ClassifiedItem` (Tasks 4/5).

- [ ] **Step 1: Write `scripts/test_color.py`**

```python
#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from ml.vision.color import dominant_color, extract_dominant_colors, rgb_to_hsv_spec


def main():
    parser = argparse.ArgumentParser(description="Extract dominant colors from an image")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--k", type=int, default=3, help="Number of color clusters")
    parser.add_argument("--min-alpha", type=int, default=128, help="Opaque alpha threshold")
    args = parser.parse_args()

    try:
        img = Image.open(args.input)
        clusters = extract_dominant_colors(img, k=args.k, min_alpha=args.min_alpha)
        print(f"Input: {args.input}")
        for i, c in enumerate(clusters):
            spec = rgb_to_hsv_spec(c.rgb)
            print(
                f"  #{i + 1} rgb={c.rgb} hue={spec.hue:.0f} "
                f"sat={spec.saturation:.2f} val={spec.value:.2f} prop={c.proportion:.2%}"
            )
        print(f"Dominant: {dominant_color(clusters)}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `scripts/test_evaluate.py`**

```python
#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.evaluate import threshold_sweep
from ml.vision.classify import ClassifiedItem

LABELS = ("top", "bottom", "dress", "outerwear", "shoes")

DEFAULT_ROWS = [
    ("t1", "top", "top", 0.90),
    ("b1", "bottom", "bottom", 0.85),
    ("b2", "bottom", "bottom", 0.80),
    ("d1", "dress", "dress", 0.90),
    ("d2", "dress", "top", 0.40),
    ("s1", "shoes", "shoes", 0.70),
    ("o1", "outerwear", "outerwear", 0.70),
]


def main():
    parser = argparse.ArgumentParser(
        description="Sweep classification confidence threshold and measure Generate impact"
    )
    parser.add_argument("--csv", help="CSV: item_id,true_category,predicted_category,confidence")
    parser.add_argument("--n-outfits", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.csv:
        with open(args.csv, newline="") as f:
            rows = list(csv.reader(f))
        items = [ClassifiedItem(r[0], r[1], r[2], float(r[3])) for r in rows]
    else:
        items = [ClassifiedItem(*r) for r in DEFAULT_ROWS]

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = threshold_sweep(
        items, LABELS, thresholds=thresholds, n_outfits=args.n_outfits, seed=args.seed
    )

    print(f"{'thr':>4} {'kept':>4} {'acc':>6} {'macF1':>6} {'valid%':>7}")
    for r in results:
        print(
            f"{r['threshold']:>4.1f} {r['kept']:>4} "
            f"{r['accuracy']:>6.3f} {r['macro_f1']:>6.3f} {r['valid_outfit_rate']:>7.2%}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the color CLI**

Run: `.venv/bin/python scripts/test_color.py --input ml/media/sample.jpg --k 3`
Expected: prints 3 cluster rows and a `Dominant:` line (no traceback).

- [ ] **Step 4: Verify the evaluate CLI**

Run: `.venv/bin/python scripts/test_evaluate.py`
Expected: prints a threshold table; `valid%` should be below 100% at `thr 0.0` and reach 100% at `thr >= 0.5` (the misclassified low-confidence dress `d2` drops out).

- [ ] **Step 5: Commit**

```bash
git add scripts/test_color.py scripts/test_evaluate.py
git commit -m "feat: add color and evaluation CLI scripts"
```

---

## Self-Review

- **Spec coverage:** Task 2 (color extraction + color-clash rule) → Tasks 1–2, 6. Task 3 (classification impact + threshold tuning) → Tasks 3–6. Task 1 (visual label check) intentionally skipped per user note.
- **Placeholders:** none — every code step has full code; commands have expected output.
- **Type/name consistency:** `ColorSpec`/`rgb_to_hsv_spec` defined in Task 1 and imported identically in Task 2; `ClassifiedItem`/`filter_by_confidence` (Task 4) and `generate_outfit`/`outfit_is_valid`/`WardrobeItem` (Task 3) are imported by `ml/evaluate.py` with matching names. `CATEGORIES`, `labels`, and `LABELS` all use the same 5-category tuple.
