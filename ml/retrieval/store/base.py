"""Vector store abstraksiyası.

Çağıran kod (search/matcher/ingest) yalnız bu interfeysi bilir — konkret
backend `config.BACKEND` ilə seçilir. numpy və pgvector backend-ləri eyni
sıralamanı qaytarmalıdır (hər ikisi exact cosine).
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.retrieval import config


@dataclass(frozen=True)
class SearchResult:
    """Bir axtarış nəticəsi. `score` = cosine similarity, [-1, 1]."""

    item_id: str
    score: float
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # CLI çıxışlarını oxunaqlı saxlamaq üçün
        cat = self.meta.get("category")
        suffix = f" ({cat})" if cat else ""
        return f"<{self.item_id} {self.score:.4f}{suffix}>"


def validate_vectors(vectors: np.ndarray, *, expect_normalized: bool = True) -> np.ndarray:
    """Embedding müqaviləsini yoxlayır: float32, [N, EMB_DIM], L2-normalized."""
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or vectors.shape[1] != config.EMB_DIM:
        raise ValueError(
            f"Vektor shape [N, {config.EMB_DIM}] olmalıdır, alındı {vectors.shape}"
        )
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError("Vektorlarda NaN/Inf var")
    if expect_normalized and len(vectors):
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(
                f"Vektorlar L2-normalized deyil (min={norms.min():.4f}, "
                f"max={norms.max():.4f}). embedder.l2_normalize() istifadə et."
            )
    return vectors


def as_query_vector(vector: np.ndarray) -> np.ndarray:
    """Sorğu vektorunu [EMB_DIM] formasına gətirir və yoxlayır."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.shape[0] != config.EMB_DIM:
        raise ValueError(
            f"Sorğu vektoru {config.EMB_DIM} ölçülü olmalıdır, alındı {vector.shape}"
        )
    return vector


def matches_filter(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Sadə bərabərlik filtri. Dəyər siyahıdırsa — «içindədir» yoxlaması."""
    if not where:
        return True
    for key, wanted in where.items():
        actual = meta.get(key)
        if isinstance(wanted, (list, tuple, set)):
            if actual not in wanted:
                return False
        elif actual != wanted:
            return False
    return True


class VectorStore(ABC):
    """Bütün backend-lərin implement etdiyi interfeys."""

    @abstractmethod
    def add(
        self,
        ids: Sequence[str],
        vecs: np.ndarray,
        meta: Sequence[dict[str, Any]] | None = None,
    ) -> int:
        """Vektorları saxlayır. Mövcud id yenilənir (upsert). Yazılan sayı qaytarır."""

    @abstractmethod
    def search(
        self,
        vec: np.ndarray,
        k: int = config.DEFAULT_TOP_K,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Exact cosine similarity ilə top-k. Azalan sıra ilə qaytarır."""

    @abstractmethod
    def count(self) -> int:
        """Saxlanılan vektor sayı."""

    @abstractmethod
    def get(self, item_id: str) -> SearchResult | None:
        """id üzrə bir yazı (vektorsuz, meta ilə). Yoxdursa None."""

    def distinct_values(self, field: str) -> list[str]:
        """Verilmiş meta sahəsindəki fərqli dəyərlər (məs. bütün kateqoriyalar).

        Matcher kateqoriya filtrini qurmaq üçün istifadə edir. Backend bunu
        dəstəkləmirsə boş siyahı qaytarılır və filtr sadəcə tətbiq olunmur.
        """
        return []

    def close(self) -> None:
        """Resursları buraxır. Default: heç nə."""

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def get_store(backend: str | None = None, **kwargs) -> VectorStore:
    """Factory — `config.BACKEND` (və ya açıq verilən ad) üzrə store qaytarır."""
    backend = (backend or config.BACKEND).lower()
    if backend == "numpy":
        from ml.retrieval.store.numpy_store import NumpyStore

        return NumpyStore(**kwargs)
    if backend in ("pg", "pgvector", "postgres"):
        from ml.retrieval.store.pg_store import PgStore

        return PgStore(**kwargs)
    raise ValueError(f"Naməlum backend: {backend!r} (gözlənilən: 'numpy' | 'pg')")


def main() -> int:
    """Interfeys və mövcud backend-lərin vəziyyətini göstərir."""
    import argparse
    import inspect

    parser = argparse.ArgumentParser(description="VectorStore interfeysi və backend-lər")
    parser.add_argument("--backends", default="numpy,pg", help="yoxlanacaq backend-lər")
    args = parser.parse_args()

    print("VectorStore abstrakt metodları:")
    for name in sorted(VectorStore.__abstractmethods__):
        signature = inspect.signature(getattr(VectorStore, name))
        print(f"  {name}{signature}")
    print(f"\nSearchResult sahələri: item_id, score, meta")
    print(f"Aktiv backend (config.BACKEND): {config.BACKEND}\n")

    for backend in (b.strip() for b in args.backends.split(",") if b.strip()):
        try:
            store = get_store(backend)
            print(f"  [OK]   {backend:<6} — {store.count()} əşya ({type(store).__name__})")
            store.close()
        except Exception as error:  # backend əlçatmaz ola bilər (məs. DB qapalı)
            print(f"  [XƏTA] {backend:<6} — {type(error).__name__}: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
