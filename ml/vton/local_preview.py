"""Offline try-on preview: put the user's garments onto the model photo locally.

The hosted try-on Spaces give the best result, but they run on a shared free
GPU with a small daily allowance, so they are unavailable much of the time.
This module is the fallback that always works: no network, no quota, about a
second per garment.

It is not a diffusion model and does not pretend to be. The approach is:

1. Segment the person photo to find where each garment belongs — the
   upper-clothes region, the trouser region, the dress region.
2. Take the wardrobe garment (already a cut-out on white) and scale it to
   cover that region.
3. Keep only the pixels inside the person's own garment silhouette, so the
   result follows the body's shape and pose, and feather the edge.

The outcome reads as "your shirt, on this person" rather than a photorealistic
render, which is the honest thing to show when the real model is unavailable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ml.vision.segmentation import segment_labels

# Which segmentation labels each garment region should replace.
REGION_LABELS: dict[str, tuple[int, ...]] = {
    "Upper-body": (4,),          # Upper-clothes
    "Lower-body": (5, 6),        # Skirt, Pants
    "Dress": (7, 4),             # Dress, falling back to upper-clothes
}
# Ignore a region the model barely found; it would only produce a smear.
MIN_REGION_RATIO = 0.01


def _garment_alpha(img: Image.Image) -> np.ndarray:
    """Opacity for a garment stored as a cut-out on a white background."""
    if img.mode == "RGBA":
        return np.asarray(img.split()[3], dtype=np.float32) / 255.0
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    # White is background; anything darker than near-white is fabric.
    return (rgb.min(axis=2) < 244).astype(np.float32)


def _fit_cover(garment: Image.Image, box_w: int, box_h: int) -> np.ndarray:
    """Scale the garment to cover a box and return opaque fabric pixels.

    The garment's own cut-out background is repainted in its dominant colour
    rather than left transparent: the preview must replace what the person is
    already wearing, and any gap would show their original clothes through
    the new ones.
    """
    scale = max(box_w / garment.width, box_h / garment.height)
    new = (max(1, round(garment.width * scale)), max(1, round(garment.height * scale)))
    resized = garment.resize(new, Image.LANCZOS)
    alpha = _garment_alpha(resized)
    rgb = np.asarray(resized.convert("RGB"), dtype=np.float32)

    fabric = alpha > 0.5
    if fabric.any():
        fill = np.median(rgb[fabric], axis=0)
        rgb = np.where(fabric[..., None], rgb, fill)

    left = max(0, (new[0] - box_w) // 2)
    top = max(0, (new[1] - box_h) // 2)
    rgb = rgb[top:top + box_h, left:left + box_w]

    pad_h, pad_w = box_h - rgb.shape[0], box_w - rgb.shape[1]
    if pad_h > 0 or pad_w > 0:
        rgb = np.pad(rgb, ((0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)), mode="edge")
    return rgb


def apply_garment(person_path: str, garment_path: str, region: str, out_path: str) -> bool:
    """Draw one garment onto the person photo. Returns False if it can't."""
    person = Image.open(person_path).convert("RGB")
    garment = Image.open(garment_path)

    seg = segment_labels(person)
    mask = np.isin(seg, REGION_LABELS.get(region, (4,)))
    if mask.mean() < MIN_REGION_RATIO:
        return False

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    box_h, box_w = y1 - y0, x1 - x0

    g_rgb = _fit_cover(garment, box_w, box_h)

    # The new garment covers exactly the region the old one occupied, so the
    # silhouette keeps the body's shape and pose.
    region_mask = mask[y0:y1, x0:x1].astype(np.float32)

    # Feather only slightly, so the edge is not cut out with scissors but the
    # old garment still does not bleed through.
    feather = max(1, int(round(min(box_h, box_w) * 0.008)))
    soft = Image.fromarray((region_mask * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(feather)
    )
    combined = np.asarray(soft, dtype=np.float32)[..., None] / 255.0

    out = np.asarray(person, dtype=np.float32).copy()
    patch = out[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = g_rgb * combined + patch * (1.0 - combined)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out.astype(np.uint8)).save(out_path)
    return True
