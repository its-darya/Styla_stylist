"""Style Classifier Inference Modulu.

Öyrədilmiş Ensemble Style MLP modelini çağıraraq verilmiş outfit
(top + bottom embedding-ləri) üçün stil təxminini və ehtimal paylanmasını qaytarır.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ml.style import config
from ml.style.dataset import StyleEncoder
from ml.style.model import EnsembleStyleMLP


@dataclass
class StylePrediction:
    """Outfit stil təxmini nəticəsi."""

    style: str
    confidence: float
    top_styles: list[tuple[str, float]]
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_tensor(vec: Union[Sequence[float], np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Embedding vektorunu 2D Float Tensoruna çevirir."""
    if isinstance(vec, torch.Tensor):
        t = vec.detach().clone().float()
    elif isinstance(vec, np.ndarray):
        t = torch.from_numpy(vec.astype(np.float32))
    elif isinstance(vec, (list, tuple)):
        t = torch.tensor(vec, dtype=torch.float32)
    else:
        t = torch.tensor(list(vec), dtype=torch.float32)

    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


class StyleClassifier:
    """Outfit-lərin stilini təsnif edən PyTorch Inference sinfi."""

    _instance: Optional[StyleClassifier] = None

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        classes_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_path = Path(model_path or config.DEFAULT_MODEL_PATH)
        self.classes_path = Path(classes_path or config.DEFAULT_CLASSES_PATH)

        self.encoder = self._load_encoder()
        self.model: Optional[EnsembleStyleMLP] = None
        self._load_model()

    def _load_encoder(self) -> StyleEncoder:
        if self.classes_path.exists():
            return StyleEncoder.load(self.classes_path)
        return StyleEncoder(config.STYLES)

    def _load_model(self) -> None:
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            cfg = checkpoint.get("config", {})
            self.model = EnsembleStyleMLP(
                num_classes=self.encoder.num_classes(),
                emb_dim=cfg.get("emb_dim", config.EMB_DIM),
                hidden_dims=cfg.get("hidden_dims", config.HIDDEN_DIMS),
                dropout=cfg.get("dropout", config.DROPOUT),
                feature_mode=cfg.get("feature_mode", config.INPUT_FEATURE_MODE),
                use_batch_norm=cfg.get("use_batch_norm", config.USE_BATCH_NORM),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            print(f"[StyleClassifier] Model yükləndi: {self.model_path}")
        else:
            print(
                f"[StyleClassifier Qeyd] Çəki faylı tapılmadı ({self.model_path}). "
                "İlkin arxitektura ilə başladılır."
            )
            self.model = EnsembleStyleMLP(
                num_classes=self.encoder.num_classes(),
                emb_dim=config.EMB_DIM,
                hidden_dims=config.HIDDEN_DIMS,
                dropout=0.0,
                feature_mode=config.INPUT_FEATURE_MODE,
                use_batch_norm=False,
            ).to(self.device)
            self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        top_emb: Union[Sequence[float], np.ndarray, torch.Tensor],
        bottom_emb: Union[Sequence[float], np.ndarray, torch.Tensor],
        top_k: int = 3,
    ) -> StylePrediction:
        """Top və Bottom embedding-ləri əsasında outfit-in stilini təxmin edir."""
        t1 = _to_tensor(top_emb).to(self.device)
        t2 = _to_tensor(bottom_emb).to(self.device)

        probs_t = self.model.predict_proba(t1, t2)[0].cpu().numpy()

        classes = self.encoder.classes
        probs_dict = {classes[i]: float(probs_t[i]) for i in range(len(classes))}

        sorted_indices = np.argsort(probs_t)[::-1]
        top_styles = [
            (classes[i], float(probs_t[i])) for i in sorted_indices[:top_k]
        ]

        best_style, best_conf = top_styles[0]

        return StylePrediction(
            style=best_style,
            confidence=best_conf,
            top_styles=top_styles,
            probabilities=probs_dict,
        )

    @torch.no_grad()
    def predict_outfit(
        self,
        item_embeddings: Sequence[Union[Sequence[float], np.ndarray, torch.Tensor]],
        top_k: int = 3,
    ) -> StylePrediction:
        """İxtiyari əşya siyahısından (məs. top, bottom, ayaqqabı) outfit stilini təyin edir."""
        if len(item_embeddings) == 0:
            raise ValueError("Ən azı 1 əşya embedding-i tələb olunur.")
        elif len(item_embeddings) == 1:
            # Tək əşya olduqda top və bottom kimi eyni vektoru göndəririk
            return self.predict(item_embeddings[0], item_embeddings[0], top_k=top_k)
        else:
            # İlk iki əsas əşyanı (məs. top və bottom) götürürük
            return self.predict(item_embeddings[0], item_embeddings[1], top_k=top_k)


# Qlobal singleton stil təsnifatçısı
_global_style_classifier: Optional[StyleClassifier] = None


def get_style_classifier(
    model_path: Optional[Union[str, Path]] = None
) -> StyleClassifier:
    global _global_style_classifier
    if _global_style_classifier is None or (
        model_path and Path(model_path) != _global_style_classifier.model_path
    ):
        _global_style_classifier = StyleClassifier(model_path=model_path)
    return _global_style_classifier


def predict_outfit_style(
    top_emb: Union[Sequence[float], np.ndarray, torch.Tensor],
    bottom_emb: Union[Sequence[float], np.ndarray, torch.Tensor],
    model_path: Optional[Union[str, Path]] = None,
    top_k: int = 3,
) -> StylePrediction:
    """Outfit-in stilini təyin edən rahat top-level funksiya.

    İstifadə:
        from ml.style import predict_outfit_style

        pred = predict_outfit_style(top_vector, bottom_vector)
        print(f"Stil: {pred.style} ({pred.confidence:.2%})")
        print(f"Top stillər: {pred.top_styles}")
    """
    classifier = get_style_classifier(model_path)
    return classifier.predict(top_emb, bottom_emb, top_k=top_k)
