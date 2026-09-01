"""Compatibility Scoring Funksiyası (PyTorch Inference).

pgvector-dan və ya digər mənbələrdən çəkilən iki geyim embedding vektoru
arasında [0, 1] aralığında uyğunluq balı hesablayır.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.compatibility import config
from ml.compatibility.model import CompatibilityMLP


def _to_tensor(vec: Union[Sequence[float], np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Müxtəlif tipli vektorları (list, numpy, torch, pgvector) 2D float32 PyTorch tensoruna çevirir."""
    if isinstance(vec, torch.Tensor):
        t = vec.detach().clone().float()
    elif isinstance(vec, np.ndarray):
        t = torch.from_numpy(vec.astype(np.float32))
    elif isinstance(vec, (list, tuple)):
        t = torch.tensor(vec, dtype=torch.float32)
    else:
        # pgvector Vector obyekti və ya iterable
        t = torch.tensor(list(vec), dtype=torch.float32)

    if t.ndim == 1:
        t = t.unsqueeze(0)  # [1, 512]
    return t


class CompatibilityScorer:
    """Öyrədilmiş Compatibility MLP modelini yükləyib sürətli inference aparan sinif."""

    _instance: Optional[CompatibilityScorer] = None

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_path = Path(model_path or config.DEFAULT_MODEL_PATH)
        self.model: Optional[CompatibilityMLP] = None
        self._load_model()

    def _load_model(self) -> None:
        """Model çəkilərini yükləyir; əgər fayl yoxdursa defolt arxitektura ilə işə düşür."""
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            cfg = checkpoint.get("config", {})
            self.model = CompatibilityMLP(
                emb_dim=cfg.get("emb_dim", config.EMB_DIM),
                hidden_dims=cfg.get("hidden_dims", config.HIDDEN_DIMS),
                dropout=cfg.get("dropout", config.DROPOUT),
                mode=cfg.get("mode", config.INPUT_REPRESENTATION),
                use_batch_norm=cfg.get("use_batch_norm", config.USE_BATCH_NORM),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            print(f"[CompatibilityScorer] Model yükləndi: {self.model_path}")
        else:
            # Fallback: model hələ öyrədilməyibsə yeni instansiya yaradılır
            print(
                f"[CompatibilityScorer Qeyd] Çəki faylı tapılmadı ({self.model_path}). "
                "İlkin model və kosinus oxşarlığı əsasında işə düşür."
            )
            self.model = CompatibilityMLP(
                emb_dim=config.EMB_DIM,
                hidden_dims=config.HIDDEN_DIMS,
                dropout=0.0,
                mode=config.INPUT_REPRESENTATION,
                use_batch_norm=False,
            ).to(self.device)
            self.model.eval()

    @torch.no_grad()
    def score(
        self,
        vec1: Union[Sequence[float], np.ndarray, torch.Tensor],
        vec2: Union[Sequence[float], np.ndarray, torch.Tensor],
    ) -> float:
        """İki geyim embedding vektoru arasında uyğunluq balı [0.0 - 1.0].

        Args:
            vec1: 1-ci geyimin pgvector/numpy/list embedding vektoru (512 ölçülü)
            vec2: 2-ci geyimin pgvector/numpy/list embedding vektoru (512 ölçülü)

        Returns:
            float: 0.0 ilə 1.0 arasında uyğunluq balı.
        """
        t1 = _to_tensor(vec1).to(self.device)
        t2 = _to_tensor(vec2).to(self.device)

        if self.model is None:
            # Fallback to normalized cosine similarity scaled to [0, 1]
            cos_sim = F.cosine_similarity(t1, t2, dim=-1).item()
            return float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))

        prob = self.model.predict_proba(t1, t2)
        score_val = float(prob.item())
        return float(np.clip(score_val, 0.0, 1.0))

    @torch.no_grad()
    def score_batch(
        self,
        vecs1: Union[np.ndarray, torch.Tensor, Sequence[Sequence[float]]],
        vecs2: Union[np.ndarray, torch.Tensor, Sequence[Sequence[float]]],
    ) -> np.ndarray:
        """Vektor partiyaları (batch) arasında uyğunluq ballarını hesablayır."""
        t1 = _to_tensor(vecs1).to(self.device)
        t2 = _to_tensor(vecs2).to(self.device)

        if self.model is None:
            cos = F.cosine_similarity(t1, t2, dim=-1).cpu().numpy()
            return np.clip((cos + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)

        probs = self.model.predict_proba(t1, t2)
        return probs.cpu().numpy().flatten().astype(np.float32)

    def score_items(
        self,
        item_id_1: str,
        item_id_2: str,
        store: Any = None,
    ) -> float:
        """PgStore və ya NumpyStore-dan id-lərə görə vektorları çəkib bal hesablayır."""
        if store is None:
            from ml.retrieval.store.numpy_store import NumpyStore
            from ml.retrieval.store.pg_store import PgStore
            try:
                store = PgStore()
            except Exception:
                store = NumpyStore()

        # Vector Store-dan vektorları əldə edirik
        if hasattr(store, "vector_of"):
            v1 = store.vector_of(item_id_1)
            v2 = store.vector_of(item_id_2)
        elif hasattr(store, "conn"):  # PgStore
            with store.conn.cursor() as cur:
                cur.execute(
                    f"SELECT embedding FROM {store.table} WHERE item_id = %s",
                    (item_id_1,),
                )
                r1 = cur.fetchone()
                cur.execute(
                    f"SELECT embedding FROM {store.table} WHERE item_id = %s",
                    (item_id_2,),
                )
                r2 = cur.fetchone()
                if not r1 or not r2:
                    raise ValueError(f"Əşya tapılmadı: {item_id_1} və ya {item_id_2}")
                v1, v2 = r1[0], r2[0]
        else:
            raise ValueError(f"Dəstəklənməyən store tipi: {type(store)}")

        if v1 is None or v2 is None:
            raise ValueError(f"Embedding tapılmadı: {item_id_1} və ya {item_id_2}")

        return self.score(v1, v2)


# Qlobal singleton scorer
_global_scorer: Optional[CompatibilityScorer] = None


def get_scorer(model_path: Optional[Union[str, Path]] = None) -> CompatibilityScorer:
    """CompatibilityScorer singleton instansiyasını qaytarır."""
    global _global_scorer
    if _global_scorer is None or (model_path and Path(model_path) != _global_scorer.model_path):
        _global_scorer = CompatibilityScorer(model_path=model_path)
    return _global_scorer


def score_compatibility(
    vec1: Union[Sequence[float], np.ndarray, torch.Tensor],
    vec2: Union[Sequence[float], np.ndarray, torch.Tensor],
    model_path: Optional[Union[str, Path]] = None,
) -> float:
    """pgvector-dan çəkilən 2 embedding vektoru arasında uyğunluq balını (0..1) qaytaran əsas funksiya.

    İstifadə:
        from ml.compatibility.scorer import score_compatibility

        score = score_compatibility(vec_top, vec_bottom)
        print(f"Uyğunluq balı: {score:.4f}")
    """
    scorer = get_scorer(model_path)
    return scorer.score(vec1, vec2)


def score_compatibility_batch(
    vecs1: Union[np.ndarray, torch.Tensor, Sequence[Sequence[float]]],
    vecs2: Union[np.ndarray, torch.Tensor, Sequence[Sequence[float]]],
    model_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Çoxlu vektor cütləri üçün uyğunluq balları massivini qaytarır."""
    scorer = get_scorer(model_path)
    return scorer.score_batch(vecs1, vecs2)
