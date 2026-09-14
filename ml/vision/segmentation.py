import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

# Laziness load
_processor = None
_model = None

LABEL_MAP = {
    "t-shirt": 4,
    "shirt": 4,
    "sweater": 4,
    "jacket": 4,
    "coat": 4,
    "hoodie": 4,
    "blazer": 4,
    "pants": 6,
    "jeans": 6,
    "shorts": 6,
    "skirt": 5,
    "dress": 7,
    "top": 4,
    "bottom": 6,
    "outerwear": 4,
}
# Labels tried in order when the category's own label finds nothing — a
# misclassified category (e.g. a skirt called "dress") should still cut out.
FALLBACK_LABELS = (4, 6, 7, 5)

def _ensure_loaded():
    global _processor, _model
    if _processor is None:
        print("[Segformer] Loading SegformerImageProcessor...")
        _processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    if _model is None:
        print("[Segformer] Loading AutoModelForSemanticSegmentation...")
        _model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        _model.eval()

def clean_mask(mask, min_ratio: float = 0.004):
    """Keep the garment's real blobs and close the gaps inside them.

    Segmenting a worn garment leaves two artefacts: small fragments where an
    arm cuts across the piece, and holes where a hand or strap sits on top of
    it. Dropping the specks removes the fragments, and filling enclosed holes
    restores the fabric behind whatever was overlapping it.
    """
    if mask.sum() == 0:
        return mask
    try:
        from scipy import ndimage
    except ImportError:  # scipy is optional — fall back to the raw mask
        return mask

    labelled, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, labelled, index=range(1, n + 1))
        biggest = sizes.max()
        keep = [i + 1 for i, s in enumerate(sizes)
                if s >= max(biggest * 0.25, mask.size * min_ratio)]
        mask = np.isin(labelled, keep)
    return ndimage.binary_fill_holes(mask)


# Body parts and accessories that sit ON TOP of clothing in a photo. Where one
# of these covers a garment, the fabric behind it is missing from the mask.
OCCLUDER_LABELS = (1, 2, 3, 8, 11, 12, 13, 14, 15, 16, 17)
# Hair, face, arms and legs — the evidence that someone is wearing the clothes.
PERSON_LABELS = (2, 11, 12, 13, 14, 15)
# Below this share of the frame there is no one in the photo, so it is already
# a garment-only product shot and must not be run through person segmentation.
PERSON_MIN_RATIO = 0.010
# A trustworthy cut-out is mostly one piece. Below this the model has carved
# the garment into unrelated scraps and the input is the better image.
COHERENCE_MIN = 0.55


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def repair_mask(mask: np.ndarray, seg: np.ndarray, close_ratio: float = 0.045):
    """Grow the garment mask back over whatever was covering it.

    An arm lying across a shirt splits the mask in two, and a hand on a hip
    punches a hole in it. Closing the mask proposes the silhouette the
    garment would have without the interruption, but closing alone would also
    bulge it outwards into the background. So a proposed pixel is only
    accepted where the photo shows a body part or accessory — i.e. something
    that could be hiding fabric — or where it is an enclosed hole.

    Returns (repaired mask, the pixels that were added).
    """
    try:
        from scipy import ndimage
    except ImportError:
        return mask, np.zeros_like(mask)

    radius = max(2, int(round(min(mask.shape) * close_ratio)))
    proposed = ndimage.binary_fill_holes(ndimage.binary_closing(mask, structure=_disk(radius)))
    added = proposed & ~mask

    occluded = np.isin(seg, OCCLUDER_LABELS)
    holes = ndimage.binary_fill_holes(mask) & ~mask
    accepted = (added & occluded) | holes
    return mask | accepted, accepted


def inpaint_region(rgb: np.ndarray, region: np.ndarray, radius: int = 4) -> np.ndarray:
    """Paint `region` in from the surrounding pixels (OpenCV Telea)."""
    if region.sum() == 0:
        return rgb
    try:
        import cv2
    except ImportError:
        return rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, region.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)


def extract_garment(image_path: str, category: str, output_path: str,
                    pad_ratio: float = 0.03, repair: bool = True) -> bool:
    """Cut the garment out of a photo and save it alone on a white canvas.

    With `repair`, fabric hidden behind an arm, a hand or a bag is rebuilt
    from the surrounding cloth instead of being left as a hole.

    Returns False when no garment is found (e.g. a photo with no person and
    no recognisable clothing), so callers can fall back to the plain
    background-removed image.
    """
    _ensure_loaded()

    cat_lower = (category or "").lower()
    label_id = LABEL_MAP.get(cat_lower, 4)
    if any(k in cat_lower for k in ("pants", "jeans", "bottom", "shorts", "trouser")):
        label_id = 6

    try:
        img = Image.open(image_path).convert("RGB")
        seg = segment_labels(img)

        # A photo with nobody in it is already a garment-only product shot.
        # This model is trained on people; run it on a flat lay and it carves
        # the garment into meaningless pieces, so leave such photos alone.
        person_ratio = float(np.isin(seg, PERSON_LABELS).mean())
        if person_ratio < PERSON_MIN_RATIO:
            print(f"[Segformer] no person in frame ({person_ratio:.3f}) — keeping image as is")
            return False

        mask = None
        for candidate in (label_id, *FALLBACK_LABELS):
            trial = clean_mask(seg == candidate)
            # Ignore specks: a real garment covers a meaningful part of the frame.
            if trial.sum() >= 0.01 * trial.size:
                mask = trial
                break
        if mask is None:
            print(f"[Segformer] no garment found (label={label_id}): {image_path}")
            return False

        rgb = np.asarray(img)
        added = np.zeros_like(mask)
        if repair:
            mask, added = repair_mask(mask, seg)
            if added.sum():
                rgb = inpaint_region(rgb, added)

        # Reject a cut-out that came out as scattered scraps rather than a
        # garment: the caller's plain background-removed image is better.
        try:
            from scipy import ndimage

            labelled, n = ndimage.label(mask)
            if n > 1:
                sizes = ndimage.sum(mask, labelled, index=range(1, n + 1))
                coherence = float(sizes.max() / sizes.sum())
                if coherence < COHERENCE_MIN:
                    print(f"[Segformer] cut-out is fragmented ({coherence:.2f}) — keeping image as is")
                    return False
        except ImportError:
            pass

        ys, xs = np.where(mask)
        pad_y, pad_x = int(img.height * pad_ratio), int(img.width * pad_ratio)
        y0, y1 = max(0, ys.min() - pad_y), min(img.height, ys.max() + 1 + pad_y)
        x0, x1 = max(0, xs.min() - pad_x), min(img.width, xs.max() + 1 + pad_x)

        crop = rgb[y0:y1, x0:x1].astype(np.float32)
        alpha = mask[y0:y1, x0:x1].astype(np.float32)[..., None]
        composed = (crop * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        Image.fromarray(composed).save(output_path)
        return True

    except Exception as e:
        print(f"[Segformer] extraction failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Outfit parsing (reference photos -> individual garment crops)
# ---------------------------------------------------------------------------

# mattmdjaga/segformer_b2_clothes label ids
SEG_LABELS = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes",
    5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe",
    11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm",
    16: "Bag", 17: "Scarf",
}
# Which segmentation labels form which outfit slot.
SLOT_OF_LABEL = {4: "top", 5: "bottom", 6: "bottom", 7: "dress"}
MAX_SIDE = 1024


@dataclass
class ParsedGarment:
    slot: str                      # top | bottom | dress
    label: str                     # Upper-clothes | Skirt | Pants | Dress
    image: Image.Image             # RGB crop, everything else painted white
    bbox: tuple[int, int, int, int]
    area_ratio: float              # garment pixels / image pixels


def segment_labels(img: Image.Image) -> np.ndarray:
    """Per-pixel label map (H x W, int) for an RGB image."""
    _ensure_loaded()
    inputs = _processor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits
    up = torch.nn.functional.interpolate(
        logits, size=img.size[::-1], mode="bilinear", align_corners=False
    )
    return up.argmax(dim=1)[0].numpy()


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected blob (drops stray mislabelled pixels)."""
    try:
        from scipy import ndimage
    except ImportError:
        return mask
    labelled, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labelled, index=range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    return labelled == keep


def parse_outfit(
    image_path: str,
    *,
    min_area_ratio: float = 0.025,
    pad_ratio: float = 0.04,
) -> list[ParsedGarment]:
    """Split a full-outfit photo into garment crops.

    Each detected slot (top / bottom / dress) becomes one crop where every
    non-garment pixel is painted white, so the crop lives in the same domain
    as the background-removed wardrobe photos it will be compared against.
    """
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > MAX_SIDE:
        scale = MAX_SIDE / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)

    seg = segment_labels(img)
    total = seg.size
    rgb = np.asarray(img)

    found: list[ParsedGarment] = []
    for label_id, slot in SLOT_OF_LABEL.items():
        mask = seg == label_id
        if mask.sum() / total < min_area_ratio:
            continue
        mask = _largest_component(mask)
        ratio = float(mask.sum()) / total
        if ratio < min_area_ratio:
            continue
        ys, xs = np.where(mask)
        pad_y = int(img.height * pad_ratio)
        pad_x = int(img.width * pad_ratio)
        y0, y1 = max(0, ys.min() - pad_y), min(img.height, ys.max() + 1 + pad_y)
        x0, x1 = max(0, xs.min() - pad_x), min(img.width, xs.max() + 1 + pad_x)

        crop = rgb[y0:y1, x0:x1].astype(np.float32)
        alpha = mask[y0:y1, x0:x1].astype(np.float32)[..., None]
        white = np.full_like(crop, 255.0)
        composed = (crop * alpha + white * (1.0 - alpha)).astype(np.uint8)
        found.append(
            ParsedGarment(
                slot=slot,
                label=SEG_LABELS[label_id],
                image=Image.fromarray(composed),
                bbox=(int(x0), int(y0), int(x1), int(y1)),
                area_ratio=ratio,
            )
        )

    # Skirt and pants are often confused on the same legs: if both appear
    # and one is tiny next to the other, keep only the larger.
    bottoms = [g for g in found if g.slot == "bottom"]
    if len(bottoms) == 2:
        big, small = sorted(bottoms, key=lambda g: g.area_ratio, reverse=True)
        if small.area_ratio < 0.4 * big.area_ratio:
            found.remove(small)

    # A small piece whose box lies almost entirely inside a much larger piece
    # is a mislabelled patch (e.g. a "skirt" on the hem of a t-shirt), not a
    # separate garment.
    def _contained(inner, outer) -> float:
        ix0, iy0, ix1, iy1 = inner.bbox
        ox0, oy0, ox1, oy1 = outer.bbox
        w = max(0, min(ix1, ox1) - max(ix0, ox0))
        h = max(0, min(iy1, oy1) - max(iy0, oy0))
        area = max(1, (ix1 - ix0) * (iy1 - iy0))
        return (w * h) / area

    for g in list(found):
        for other in found:
            if other is g or other.area_ratio <= g.area_ratio:
                continue
            if g.area_ratio < 0.3 * other.area_ratio and _contained(g, other) >= 0.7:
                found.remove(g)
                break

    order = {"top": 0, "dress": 1, "bottom": 2}
    found.sort(key=lambda g: order.get(g.slot, 9))
    return found
