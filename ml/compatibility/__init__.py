"""Styla · Compatibility & Ranking Modulu (Rol A).

Geyim cütlərinin vizual və stil uyğunluğunu (compatibility) qiymətləndirən
PyTorch modelləri, təlim skriptləri və sürətli scoring funksiyaları.

İctimai API:
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
]
