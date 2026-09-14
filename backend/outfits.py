"""Outfit generation for a user's wardrobe.

Pipeline (all per request, ~100 items => a few thousand candidates, < 1 s):

1. Load the user's items with embeddings and slot them: top / bottom /
   dress / outerwear (shoes and accessories are ignored).
2. Build candidates: (top, bottom) and (dress,), each optionally with one
   outerwear piece when the style allows it. Gender-consistent only.
3. Apply the style's hard `deny_*` rules — but if that would leave nothing,
   fall back to soft penalties so the user still gets an outfit.
4. Score every candidate on four signals:
     - colour        : an absolute rule over the whole outfit's palette —
                       how many pieces make a colour statement, and how
                       those hues relate. Deliberately NOT normalised across
                       candidates, or the best of an all-clashing set would
                       be rescaled back up to a perfect score.
     - style         : zero-shot CLIP similarity to the requested style,
                       centred across the wardrobe so "casual" doesn't win
                       everything, then min-max normalised
     - compatibility : the compatibility model, or CLIP cosine when no
                       checkpoint is trained. Weighted least, because cosine
                       measures how similar two garments look rather than
                       whether they belong together
     - personal      : similarity to the user's own style reference photos
                       (only when they uploaded some)
   plus `rules`: pattern clash, preferred/penalised colours, patterns and
   categories from style_rules.json. Looks whose colours fight are dropped
   outright while enough clean ones remain.
5. Pick the top N with a diversity penalty so "Regenerate" cycles through
   genuinely different looks instead of the same shirt with five bottoms.
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ml.compatibility.rules import pattern_clash
from ml.retrieval import config as rconfig
from ml.retrieval.style_prompts import load_style_embeddings
from ml.retrieval.style_scorer import StyleScorer

BASE_DIR = Path(__file__).resolve().parent.parent
STYLE_RULES_PATH = BASE_DIR / "ml" / "compatibility" / "style_rules.json"

# Frontend style id -> prompt text for zero-shot style scoring.
STYLE_PROMPTS: dict[str, str] = {
    "casual": "casual",
    "formal": "formal",
    "business-casual": "business casual",
    "streetwear": "streetwear",
    "sporty": "sporty athletic",
    "bohemian": "bohemian boho",
    "minimalist": "minimalist",
    "elegant": "elegant evening",
}

SLOT_TOP = ("t-shirt", "shirt", "sweater")
SLOT_BOTTOM = ("pants", "jeans", "shorts", "skirt")
SLOT_DRESS = ("dress",)
SLOT_OUTER = ("jacket", "coat")

# Colour-name -> (hue°, saturation, value). Saturation < 0.15 counts as neutral.
COLOR_HSV: dict[str, tuple[float, float, float]] = {
    "black": (0, 0.0, 0.05), "white": (0, 0.0, 1.0), "gray": (0, 0.0, 0.5),
    "grey": (0, 0.0, 0.5), "beige": (40, 0.12, 0.88), "red": (0, 0.9, 0.8),
    "blue": (220, 0.85, 0.8), "green": (120, 0.7, 0.6), "yellow": (55, 0.9, 0.95),
    "orange": (30, 0.9, 0.95), "purple": (280, 0.7, 0.6), "pink": (340, 0.5, 0.95),
    "brown": (25, 0.6, 0.45), "navy": (230, 0.8, 0.3), "maroon": (350, 0.8, 0.4),
    "black & white": (0, 0.0, 0.5), "navy & white": (230, 0.5, 0.6),
    "blue & white": (220, 0.5, 0.85), "red & white": (0, 0.55, 0.9),
}

# Colours that go with anything, so they never count as a "statement".
NEUTRALS = {
    "black", "white", "gray", "grey", "beige", "navy", "brown", "cream",
    "black & white", "navy & white",
}
# A colour below this saturation reads as a neutral even if not listed above.
NEUTRAL_SATURATION = 0.2

DEFAULT_COUNT = 8
DIVERSITY_PENALTY = 0.10   # per item shared with an already-selected outfit
LOOKALIKE_PENALTY = 0.25   # per earlier look with the same colour/type wording
# Below this the colours actively fight; such looks are dropped when there are
# enough clean ones to show instead.
COLOR_CLASH_MAX = 0.35
# Displayed "match %" = DISPLAY_FLOOR + DISPLAY_SPAN * base + rule bonuses, capped.
DISPLAY_FLOOR, DISPLAY_SPAN, DISPLAY_CAP = 0.25, 0.70, 0.99
# How much of the style rule bonus reaches the displayed percentage.
DISPLAY_RULES_SHARE = 0.25

# Colour leads, because it is what makes an outfit read as deliberate, and it
# is the one signal measured on an absolute scale. The compatibility model is
# weighted least: with no trained checkpoint it falls back to CLIP cosine,
# which measures how similar two garments look, not whether they go together.
W_COLOR, W_STYLE, W_COMPAT = 0.45, 0.35, 0.20
W_COLOR_P, W_STYLE_P, W_COMPAT_P, W_PERSONAL_P = 0.34, 0.26, 0.15, 0.25


@dataclass
class Item:
    id: str
    image_url: str
    category: str
    color: str
    pattern: str
    gender: str
    embedding: np.ndarray
    date_added: str | None = None

    @property
    def slot(self) -> str | None:
        c = (self.category or "").lower()
        if c in SLOT_TOP:
            return "top"
        if c in SLOT_BOTTOM:
            return "bottom"
        if c in SLOT_DRESS:
            return "dress"
        if c in SLOT_OUTER:
            return "outerwear"
        return None

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "imageUrl": self.image_url,
            "category": self.category,
            "color": self.color,
            "pattern": self.pattern,
            "gender": self.gender,
            "dateAdded": self.date_added,
        }


@dataclass
class Candidate:
    items: tuple[Item, ...]
    compat: float = 0.0
    color: float = 0.0
    style: float = 0.0
    personal: float = 0.0
    rules: float = 0.0
    denied: int = 0
    total: float = 0.0
    display: float = 0.0
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Colour harmony
# ---------------------------------------------------------------------------

def _hue_distance(h1: float, h2: float) -> float:
    d = abs(h1 - h2) % 360.0
    return min(d, 360.0 - d)


def is_neutral(color: str) -> bool:
    """Black, white, grey, beige, navy, brown — the colours that go with all."""
    c = (color or "").strip().lower()
    if c in NEUTRALS:
        return True
    hsv = COLOR_HSV.get(c)
    return hsv is not None and hsv[1] < NEUTRAL_SATURATION


def outfit_colors(items: Sequence[Item]) -> tuple[float, str]:
    """Score a whole outfit's palette in [0, 1], with a reason.

    This follows how outfits are actually put together rather than averaging
    pairs: count the pieces that make a colour statement, and judge the look
    on how those relate. One statement against neutrals is the classic and
    scores highest; two statements are fine when their hues sit close
    together or straight opposite, and poor in between; three fight each
    other. Averaging pairs instead let a bad pairing hide behind good ones.
    """
    colors = [(i.color or "").strip().lower() for i in items]
    loud = [c for c in colors if not is_neutral(c)]

    if not loud:
        return 0.86, "neutral palette"
    if len(loud) == 1:
        if len(colors) == 1:
            return 0.86, f"{loud[0]} on its own"
        return 1.0, f"{loud[0]} against neutrals"
    if len(loud) > 2:
        return 0.15, f"{len(loud)} colours competing"

    first, second = loud[0], loud[1]
    # A multi-coloured piece is already a statement; a second one is a fight.
    if "multi" in first or "multi" in second:
        return 0.28, "two busy pieces"
    if first == second:
        return 0.88, f"tonal {first}"

    ha, hb = COLOR_HSV.get(first), COLOR_HSV.get(second)
    if ha is None or hb is None:
        return 0.6, f"{first} + {second}"
    d = _hue_distance(ha[0], hb[0])
    if d <= 30:
        return 0.84, f"{first} + {second} tonal"
    if d <= 60:
        return 0.76, f"{first} + {second} analogous"
    if d >= 150:
        return 0.68, f"{first} + {second} complementary"
    return 0.18, f"{first} + {second} clash"


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def load_style_rules() -> dict[str, dict[str, Any]]:
    try:
        with open(STYLE_RULES_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # rules are optional; generation still works
        print(f"[outfits] could not load style rules: {exc}")
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _lower_set(values: Sequence[str] | None) -> set[str]:
    return {str(v).lower() for v in (values or [])}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class OutfitGenerator:
    def __init__(self, embedder, compat_scorer, personal_style=None) -> None:
        self.compat_scorer = compat_scorer
        self.personal_style = personal_style
        style_embs = load_style_embeddings(embedder, styles=list(STYLE_PROMPTS.values()))
        self.style_scorer = StyleScorer(style_embs=style_embs, embedder=embedder)
        self.rules = load_style_rules()

    # --- candidates -------------------------------------------------------
    @staticmethod
    def _gender_ok(items: Sequence[Item], wanted: str) -> bool:
        genders = {(i.gender or "unisex").lower() for i in items}
        specific = genders - {"unisex", ""}
        if len(specific) > 1:
            return False
        if wanted not in ("", "any") and specific and wanted not in specific:
            return False
        return True

    def _candidates(self, items: list[Item], rule: dict[str, Any], gender: str,
                    outerwear: str = "auto") -> list[Candidate]:
        tops = [i for i in items if i.slot == "top"]
        bottoms = [i for i in items if i.slot == "bottom"]
        dresses = [i for i in items if i.slot == "dress"]
        outers = [i for i in items if i.slot == "outerwear"]
        # "auto" defers to the style; the caller can force it either way.
        outer_mode = rule.get("outerwear", "optional") if outerwear == "auto" else (
            "preferred" if outerwear == "always" else "never"
        )

        bases: list[tuple[Item, ...]] = [(t, b) for t, b in itertools.product(tops, bottoms)]
        if rule.get("allow_dress", True):
            bases.extend((d,) for d in dresses)

        # Layering over a dress only works for some styles, and then only in a
        # neutral: a blazer over a shift dress reads as deliberate, a coloured
        # jacket over a patterned dress reads as an accident.
        layer_over_dress = rule.get("layer_over_dress", False)
        dress_outers = [o for o in outers if is_neutral(o.color)] if layer_over_dress else []

        combos: list[tuple[Item, ...]] = []
        for base in bases:
            is_dress = len(base) == 1 and base[0].slot == "dress"
            allowed_outers = dress_outers if is_dress else outers
            # A dress is a complete outfit, so it is always offered bare.
            if outer_mode != "preferred" or not allowed_outers or is_dress:
                combos.append(base)
            if outer_mode != "never":
                combos.extend(base + (o,) for o in allowed_outers)
        return [Candidate(items=c) for c in combos if self._gender_ok(c, gender)]

    # --- scoring ----------------------------------------------------------
    def _compat_scores(self, cands: list[Candidate]) -> None:
        """Score the palette (absolute) and the compatibility model (relative)."""
        for c in cands:
            c.color, note = outfit_colors(c.items)
            if note:
                c.notes.append(note)

        pairs: list[tuple[int, Item, Item]] = []
        for ci, c in enumerate(cands):
            for a, b in itertools.combinations(c.items, 2):
                pairs.append((ci, a, b))
        if not pairs:
            return

        v1 = np.stack([a.embedding for _, a, _ in pairs])
        v2 = np.stack([b.embedding for _, _, b in pairs])
        model_part = _min_max(
            np.asarray(self.compat_scorer.score_batch(v1, v2), dtype=np.float32)
        )
        per_cand: dict[int, list[float]] = {}
        for k, (ci, _, _) in enumerate(pairs):
            per_cand.setdefault(ci, []).append(float(model_part[k]))
        for ci, c in enumerate(cands):
            scores = per_cand.get(ci)
            # A dress on its own has no pair to compare: treat it as neutral.
            c.compat = float(np.mean(scores)) if scores else 0.5

    def _style_scores(self, cands: list[Candidate], items: list[Item], style_id: str) -> None:
        prompt = STYLE_PROMPTS.get(style_id, style_id)
        embs = np.stack([i.embedding for i in items])
        res = self.style_scorer.score_styles(embs, centering=len(items) >= 2)
        if prompt not in res["styles"]:
            for c in cands:
                c.style = 0.0
            return
        col = res["styles"].index(prompt)
        per_item = {i.id: float(res["centered"][k][col]) for k, i in enumerate(items)}
        for c in cands:
            c.style = float(np.mean([per_item[i.id] for i in c.items]))

    def _personal_scores(self, cands: list[Candidate], items: list[Item], user_id: str) -> bool:
        if self.personal_style is None or self.personal_style.count(user_id) == 0:
            return False
        embs = np.stack([i.embedding for i in items])
        scores = self.personal_style.personal_score(embs, user_id)
        per_item = {i.id: float(scores[k]) for k, i in enumerate(items)}
        for c in cands:
            c.personal = float(np.mean([per_item[i.id] for i in c.items]))
        return True

    @staticmethod
    def _rule_scores(cands: list[Candidate], rule: dict[str, Any]) -> None:
        deny_cat = _lower_set(rule.get("deny_categories"))
        deny_pat = _lower_set(rule.get("deny_patterns"))
        pen_cat = _lower_set(rule.get("penalty_categories"))
        pref_cat = _lower_set(rule.get("prefer_categories"))
        pref_col = _lower_set(rule.get("prefer_colors"))
        pref_pat = _lower_set(rule.get("prefer_patterns"))

        for c in cands:
            score = 0.0
            denied = 0
            for it in c.items:
                cat, pat, col = it.category.lower(), (it.pattern or "").lower(), (it.color or "").lower()
                if cat in deny_cat or pat in deny_pat:
                    denied += 1
                if cat in pen_cat:
                    score -= 0.08
                if cat in pref_cat:
                    score += 0.05
                if col in pref_col:
                    score += 0.05
                if pat in pref_pat:
                    score += 0.03
            if pattern_clash([it.pattern or "Solid" for it in c.items]):
                score -= 0.15
                c.notes.append("pattern clash")
            c.rules = score
            c.denied = denied

    # --- selection --------------------------------------------------------
    @staticmethod
    def _select_diverse(cands: list[Candidate], count: int, max_outer: int | None = None) -> list[Candidate]:
        """Pick `count` looks, favouring high scores but avoiding repetition.

        Three forces work against a monotonous result set: reusing a garment
        costs DIVERSITY_PENALTY per repeat; repeating a look's *description*
        costs LOOKALIKE_PENALTY, because two different navy shirts over the
        same jeans read as the same outfit to the person looking at it; and
        `max_outer` caps how many of the chosen looks may include an
        outerwear piece. Without that cap a wardrobe with many jackets sweeps
        the top of the ranking, since every (top, bottom) pair also exists as
        a three-piece candidate.
        """
        remaining = sorted(cands, key=lambda c: c.total, reverse=True)
        chosen: list[Candidate] = []
        used: dict[str, int] = {}
        looks: dict[str, int] = {}
        outer_used = 0

        def has_outer(c: Candidate) -> bool:
            return any(it.slot == "outerwear" for it in c.items)

        def signature(c: Candidate) -> str:
            return "|".join(sorted(f"{i.color.lower()} {i.category.lower()}" for i in c.items))

        while remaining and len(chosen) < count:
            best_i, best_v = None, -1e9
            for i, c in enumerate(remaining):
                if max_outer is not None and outer_used >= max_outer and has_outer(c):
                    continue
                shared = sum(used.get(it.id, 0) for it in c.items)
                v = (c.total
                     - DIVERSITY_PENALTY * shared
                     - LOOKALIKE_PENALTY * looks.get(signature(c), 0))
                if v > best_v:
                    best_i, best_v = i, v
            if best_i is None:  # only outerwear looks left — take the best of them
                best_i = 0
            pick = remaining.pop(best_i)
            chosen.append(pick)
            if has_outer(pick):
                outer_used += 1
            looks[signature(pick)] = looks.get(signature(pick), 0) + 1
            for it in pick.items:
                used[it.id] = used.get(it.id, 0) + 1
        return chosen

    # --- public -----------------------------------------------------------
    def generate(
        self,
        items: list[Item],
        style_id: str,
        *,
        gender: str = "any",
        user_id: str | None = None,
        use_personal_style: bool = False,
        count: int = DEFAULT_COUNT,
        outerwear: str = "auto",
        must_include: str | None = None,
        exclude_ids: Sequence[str] | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Rank outfits for a style.

        `must_include` builds every look around one garment the user picked,
        `exclude_ids` drops pieces they rejected, `outerwear` forces a layer on
        or off, and `offset` skips the looks already shown so asking again
        returns genuinely new combinations rather than the same ranking.
        """
        style_id = (style_id or "casual").lower()
        gender = (gender or "any").lower()
        rule = self.rules.get(style_id, {})

        usable = [i for i in items if i.slot is not None]
        excluded = {str(x) for x in (exclude_ids or [])}
        if excluded:
            usable = [i for i in usable if i.id not in excluded]

        anchor = next((i for i in usable if i.id == must_include), None) if must_include else None
        if must_include and anchor is None:
            return []  # the anchor was excluded or is not wearable

        if gender not in ("", "any"):
            # The anchor is the user's explicit choice, so it overrides the filter.
            usable = [i for i in usable
                      if (i.gender or "unisex").lower() in (gender, "unisex", "") or i is anchor]
        if not usable:
            return []

        cands = self._candidates(usable, rule, gender, outerwear)
        if anchor is not None:
            cands = [c for c in cands if any(i.id == anchor.id for i in c.items)]
        if not cands:
            return []

        self._rule_scores(cands, rule)
        allowed = [c for c in cands if c.denied == 0]
        if allowed:
            cands = allowed
        else:  # nothing survives the hard rules: keep everything, penalise instead
            for c in cands:
                c.rules -= 0.2 * c.denied

        self._compat_scores(cands)

        # Colours that fight are not worth showing while clean looks exist.
        clean = [c for c in cands if c.color > COLOR_CLASH_MAX]
        if len(clean) >= count:
            cands = clean

        self._style_scores(cands, usable, style_id)
        has_personal = bool(use_personal_style and user_id) and self._personal_scores(cands, usable, user_id)

        # Colour stays on its own absolute scale — min-max would rescale the
        # best of an all-clashing set back up to a perfect score.
        compat = np.array([c.compat for c in cands], dtype=np.float32)
        color = np.array([c.color for c in cands], dtype=np.float32)
        style = _min_max(np.array([c.style for c in cands]))
        personal = _min_max(np.array([c.personal for c in cands])) if has_personal else None
        for k, c in enumerate(cands):
            if personal is not None:
                base = (W_COLOR_P * color[k] + W_STYLE_P * style[k]
                        + W_COMPAT_P * compat[k] + W_PERSONAL_P * personal[k])
            else:
                base = W_COLOR * color[k] + W_STYLE * style[k] + W_COMPAT * compat[k]
            c.total = float(base + c.rules)
            # The style bonuses steer the ranking, but letting them into the
            # displayed number pushed most looks against the cap, so the
            # percentage stopped telling them apart.
            c.display = float(
                np.clip(DISPLAY_FLOOR + DISPLAY_SPAN * base + DISPLAY_RULES_SHARE * c.rules,
                        0.0, DISPLAY_CAP)
            )

        count = max(1, count)
        offset = max(0, int(offset))
        wanted = count + offset
        # Styles that call for a layer may use it throughout; elsewhere keep
        # outerwear to at most half the looks so plain pairs get a showing.
        forced = outerwear if outerwear != "auto" else rule.get("outerwear", "optional")
        max_outer = wanted if forced in ("preferred", "always") else (wanted + 1) // 2
        chosen = self._select_diverse(cands, wanted, max_outer)[offset:]
        now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        results = []
        for c in chosen:
            k = cands.index(c)
            results.append(
                {
                    "id": str(uuid.uuid4()),
                    "style": style_id,
                    "items": [it.to_public() for it in c.items],
                    "createdAt": now,
                    "score": round(c.display, 3),
                    "breakdown": {
                        "color": round(float(color[k]), 3),
                        "compatibility": round(float(compat[k]), 3),
                        "style": round(float(style[k]), 3),
                        "personal": round(float(personal[k]), 3) if personal is not None else None,
                        "rules": round(float(c.rules), 3),
                    },
                    "notes": c.notes[:3],
                    "summary": _summarise(c),
                }
            )
        return results


def _summarise(c: Candidate) -> str:
    """One line naming the pieces, e.g. "navy shirt with grey jeans"."""
    names = [f"{it.color.lower()} {it.category.lower()}" for it in c.items]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} with {names[1]}"
    return f"{names[0]} with {names[1]}, layered under {names[-1]}"


def _min_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-6:
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


def row_to_item(row: Sequence[Any], image_url: str) -> Item:
    """`SELECT item_id, category, color, pattern, gender, embedding, created_at` -> Item."""
    emb = row[5]
    if isinstance(emb, str):
        emb = np.array(emb.strip("[]").split(","), dtype=np.float32)
    elif hasattr(emb, "to_numpy"):
        emb = emb.to_numpy()
    emb = np.asarray(emb, dtype=np.float32)
    return Item(
        id=row[0],
        image_url=image_url,
        category=row[1] or "",
        color=row[2] or "Unknown",
        pattern=row[3] or "Solid",
        gender=row[4] or "unisex",
        embedding=emb,
        date_added=row[6].isoformat() if row[6] else None,
    )
