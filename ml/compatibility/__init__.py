"""Styla A Compatibility & Ranking Modulu (Rol A).

Geyim cAtlTrinin vizual vT stil uyYunluYunu (compatibility) qiymTtlTndirTn
PyTorch modellTri, tTlim skriptlTri vT sArTtli scoring funksiyalar.

ctimai API:
    from ml.compatibility import (
        CompatibilityMLP,
        CompatibilityScorer,
        score_compatibility,
        score_compatibility_batch,
        train_compatibility_model,
    )
"""
from ml.compatibility import config
from ml.compatibility.model import CompatibilityMLP, TypeAwareCompatibilityModel
from ml.compatibility.dataset import (
    PairCompatibilityDataset,
    create_pairs_from_outfits,
    load_pairs_from_npz,
    save_pairs_to_npz,
    split_dataset,
    get_dataloaders,
)
from ml.compatibility.scorer import (
    CompatibilityScorer,
    get_scorer,
    score_compatibility,
    score_compatibility_batch,
)
from ml.compatibility.train import train_compatibility_model
from ml.compatibility.generate import (
    CATEGORIES,
    Outfit,
    WardrobeItem,
    generate_outfit,
    outfit_is_valid,
)
from ml.compatibility.rules import color_clash, hue_distance, is_neutral, pattern_clash

__all__ = [
    "config",
    "CompatibilityMLP",
    "TypeAwareCompatibilityModel",
    "PairCompatibilityDataset",
    "create_pairs_from_outfits",
    "load_pairs_from_npz",
    "save_pairs_to_npz",
    "split_dataset",
    "get_dataloaders",
    "CompatibilityScorer",
    "get_scorer",
    "score_compatibility",
    "score_compatibility_batch",
    "train_compatibility_model",
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
