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
