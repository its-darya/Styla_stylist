"""Reference-look matching: "can I recreate this outfit from my wardrobe?"

The old implementation embedded the *whole* reference photo (a person, a
background, several garments) as one vector and compared it with single
garment photos — the two live in different visual domains, so scores were
low and often wrong. This version follows the README pipeline:

1. Parse the reference photo into garments with the clothes segmentation
   model (`ml/vision/segmentation.parse_outfit`): upper-clothes, pants /
   skirt, dress. Every crop has non-garment pixels painted white, matching
   the background-removed wardrobe photos.
2. Embed each crop with FashionCLIP and describe it (zero-shot colour,
   pattern, fine category restricted to the crop's slot).
3. Search only the user's wardrobe items in the same slot, re-rank with
   colour / pattern agreement, and decide matched vs. missing per piece.
4. For missing pieces, return shop-search links built from the detected
   description (e.g. "black skirt").

If segmentation finds no garment (flat-lay photo of a single item), the
whole image is background-removed and treated as one piece.
"""
from __future__ import annotations

import urllib.parse
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ml.retrieval.store.base import SearchResult
from ml.vision.background import remove_background
from ml.vision.segmentation import ParsedGarment, parse_outfit

SLOT_CATEGORIES: dict[str, list[str]] = {
    "top": ["t-shirt", "shirt", "sweater", "jacket", "coat"],
    "bottom": ["pants", "jeans", "shorts", "skirt"],
    "dress": ["dress"],
}
LABEL_CATEGORIES: dict[str, list[str]] = {
    "Skirt": ["skirt"],
    "Pants": ["pants", "jeans", "shorts"],
}

# Cosine similarity (crop <-> wardrobe photo) at/above which a piece counts as
# owned. Below it the piece is reported as missing, with the closest item
# shown for context.
MATCH_THRESHOLD = 0.58
# Display calibration: cosine 0.35 -> 0 %, 0.95 -> 100 %. Identical garments
# photographed differently score ~0.85-0.95; unrelated garments ~0.4-0.5.
DISPLAY_LO, DISPLAY_HI = 0.35, 0.95
COLOR_BOOST = 0.06
PATTERN_BOOST = 0.03
TOP_K = 6
ALTERNATES = 2

SHOP_LINKS = [
    ("Google Shopping", "https://www.google.com/search?tbm=shop&q={q}"),
    ("ASOS", "https://www.asos.com/search/?q={q}"),
    ("Zara", "https://www.zara.com/us/en/search?searchTerm={q}"),
]


def to_percent(cosine: float) -> int:
    x = (cosine - DISPLAY_LO) / (DISPLAY_HI - DISPLAY_LO)
    return int(round(100 * float(np.clip(x, 0.0, 1.0))))


ACCESSORY_CATEGORIES = ("bag", "hat", "scarf", "sunglasses", "watch", "belt")


def coarse_category(fine: str) -> str:
    c = (fine or "").lower()
    if c in SLOT_CATEGORIES["bottom"]:
        return "bottom"
    if c == "dress":
        return "dress"
    if c in ("jacket", "coat"):
        return "outerwear"
    if c in ("shoes", "sneakers", "boots", "heels", "sandals"):
        return "shoes"
    if c in ACCESSORY_CATEGORIES:
        return "accessory"
    return "top"


class ReferenceMatcher:
    def __init__(
        self,
        *,
        store,
        embedder,
        category_classifier,
        color_classifier,
        pattern_classifier,
        crops_dir: Path,
        public_url,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.categories = category_classifier
        self.colors = color_classifier
        self.patterns = pattern_classifier
        self.crops_dir = crops_dir
        self.public_url = public_url  # callable: "/data/..." -> absolute URL
        self.crops_dir.mkdir(parents=True, exist_ok=True)

    # --- parsing ------------------------------------------------------------
    def _pieces(self, image_path: Path) -> list[ParsedGarment]:
        try:
            pieces = parse_outfit(str(image_path))
        except Exception as exc:
            print(f"[reference] segmentation failed, using whole image: {exc}")
            pieces = []
        if pieces:
            return pieces

        # Fallback: a single garment photo (flat lay / product shot).
        tmp_png = image_path.with_suffix(".cut.png")
        try:
            remove_background(image_path, tmp_png, background="white")
            img = Image.open(tmp_png).convert("RGB")
        except Exception:
            img = Image.open(image_path).convert("RGB")
        finally:
            if tmp_png.exists():
                tmp_png.unlink()
        vec = self.embedder.embed_pil_images([img])[0]
        fine, _ = self.categories.classify_vector(vec)
        slot = coarse_category(fine)
        slot = "top" if slot == "outerwear" else slot
        return [ParsedGarment(slot=slot, label="Whole image", image=img,
                              bbox=(0, 0, img.width, img.height), area_ratio=1.0)]

    def _save_crop(self, img: Image.Image, slot: str) -> str:
        name = f"{uuid.uuid4().hex}_{slot}.png"
        img.save(self.crops_dir / name, "PNG")
        return self.public_url(f"/data/reference/{name}")

    # --- wardrobe lookup ------------------------------------------------------
    def _search(self, vec: np.ndarray, user_id: str, categories: list[str]) -> list[SearchResult]:
        return self.store.search(vec, k=TOP_K, where={"user_id": user_id, "category": categories})

    def _wardrobe_item(self, r: SearchResult) -> dict[str, Any]:
        meta = r.meta
        return {
            "id": r.item_id,
            "imageUrl": self.public_url(meta.get("image_path") or ""),
            "category": coarse_category(meta.get("category")),
            "fineCategory": meta.get("category") or "",
            "color": meta.get("color") or "Unknown",
            "pattern": meta.get("pattern") or "Solid",
            "gender": meta.get("gender") or "unisex",
            "dateAdded": "",
        }

    # --- main -----------------------------------------------------------------
    def match(self, image_path: Path, user_id: str, source_url: str = "") -> dict[str, Any]:
        pieces = self._pieces(image_path)
        vectors = self.embedder.embed_pil_images([p.image for p in pieces])

        matched: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        parsed: list[dict[str, Any]] = []

        for piece, vec in zip(pieces, vectors):
            allowed = LABEL_CATEGORIES.get(piece.label) or SLOT_CATEGORIES[piece.slot]
            fine, fine_conf = self.categories.classify_among(vec, allowed)
            color, _ = self.colors.classify_vector(vec)
            pattern, _ = self.patterns.classify_vector(vec)
            crop_url = self._save_crop(piece.image, piece.slot)
            detected = {"category": fine, "color": color, "pattern": pattern,
                        "confidence": round(fine_conf, 3)}
            parsed.append({"slot": piece.slot, "label": piece.label, "imageUrl": crop_url,
                           "areaRatio": round(piece.area_ratio, 3), **detected})

            results = self._search(vec, user_id, allowed)
            if not results and allowed != SLOT_CATEGORIES[piece.slot]:
                results = self._search(vec, user_id, SLOT_CATEGORIES[piece.slot])

            ranked = []
            for r in results:
                bonus = 0.0
                if (r.meta.get("color") or "").lower() == color.lower() and color != "multi-color":
                    bonus += COLOR_BOOST
                if (r.meta.get("pattern") or "").lower() == pattern.lower():
                    bonus += PATTERN_BOOST
                ranked.append((float(r.score) + bonus, float(r.score), r))
            ranked.sort(key=lambda t: t[0], reverse=True)

            if ranked and ranked[0][1] >= MATCH_THRESHOLD:
                best = ranked[0]
                matched.append(
                    {
                        "slot": piece.slot,
                        "referenceImageUrl": crop_url,
                        "detected": detected,
                        "wardrobeItem": self._wardrobe_item(best[2]),
                        "matchScore": to_percent(best[1]),
                        "alternates": [
                            {"wardrobeItem": self._wardrobe_item(r), "matchScore": to_percent(cos)}
                            for _, cos, r in ranked[1:1 + ALTERNATES]
                        ],
                    }
                )
            else:
                closest = None
                if ranked:
                    closest = {"wardrobeItem": self._wardrobe_item(ranked[0][2]),
                               "matchScore": to_percent(ranked[0][1])}
                query = f"{color} {fine}".strip()
                missing.append(
                    {
                        "slot": piece.slot,
                        "referenceImageUrl": crop_url,
                        "category": piece.slot if piece.slot != "top" else coarse_category(fine),
                        "detected": detected,
                        "closest": closest,
                        "suggestedProducts": [
                            {
                                "name": query.title(),
                                "store": store_name,
                                "price": "",
                                "imageUrl": crop_url,
                                "url": url.format(q=urllib.parse.quote_plus(query)),
                            }
                            for store_name, url in SHOP_LINKS
                        ],
                    }
                )

        total = len(pieces)
        return {
            "sourceImageUrl": source_url,
            "pieces": parsed,
            "matchedItems": matched,
            "missingItems": missing,
            "coverage": round(len(matched) / total, 3) if total else 0.0,
        }
