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
