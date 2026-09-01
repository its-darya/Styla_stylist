"""Styla · Ensemble-Level Style Classifier Modulu.

Outfit-lərin (top+bottom və ya bütün əşyalar) stilini (casual, streetwear,
formal, sporty və s.) təsnif edən PyTorch modelləri və inference funksiyaları.

İctimai API:
    from ml.style import (
        EnsembleStyleMLP,
        StyleClassifier,
        predict_outfit_style,
        train_style_model,
    )
"""
from ml.style import config
from ml.style.model import EnsembleStyleMLP, MultiItemStyleClassifier
from ml.style.dataset import (
    OutfitStyleDataset,
    StyleEncoder,
    load_style_data_from_npz,
    save_style_data_to_npz,
    generate_pseudo_labels_with_clip,
    split_style_dataset,
)
from ml.style.classifier import (
    StylePrediction,
    StyleClassifier,
    get_style_classifier,
    predict_outfit_style,
)
from ml.style.train import train_style_model

__all__ = [
    "config",
    "EnsembleStyleMLP",
    "MultiItemStyleClassifier",
    "OutfitStyleDataset",
    "StyleEncoder",
    "StylePrediction",
    "StyleClassifier",
    "get_style_classifier",
    "predict_outfit_style",
    "train_style_model",
    "load_style_data_from_npz",
    "save_style_data_to_npz",
    "generate_pseudo_labels_with_clip",
    "split_style_dataset",
]
