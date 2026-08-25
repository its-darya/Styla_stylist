"""numpy backend — `.npy` + `ids.json`, exact cosine search.

Disk formatı (A modulu bunu DB-siz oxuyur, dəyişdirmə):
    embeddings.npy : float32, shape [N, EMB_DIM], L2-normalized
    ids.json       : {"ids": [...], "meta": {id: {...}}, "dim": 512, ...}
    ids[i] <-> embeddings[i] — sıra eyni indeksdədir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.retrieval import config
from ml.retrieval.store.base import (
    SearchResult,
    VectorStore,
    as_query_vector,
    matches_filter,
    validate_vectors,
)


class NumpyStore(VectorStore):
    def __init__(
        self,
        emb_path: str | Path = config.EMB_PATH,
        ids_path: str | Path = config.IDS_PATH,
        autoload: bool = True,
    ) -> None:
        self.emb_path = Path(emb_path)
        self.ids_path = Path(ids_path)
        self._vectors = np.zeros((0, config.EMB_DIM), dtype=np.float32)
        self._ids: list[str] = []
        self._meta: dict[str, dict[str, Any]] = {}
        if autoload:
            self.load()

    # --- persistence ------------------------------------------------------
    def load(self) -> bool:
        """Diskdən oxuyur. Fayllar yoxdursa boş store ilə davam edir."""
        if not (self.emb_path.exists() and self.ids_path.exists()):
            return False
        vectors = np.load(self.emb_path)
        payload = json.loads(self.ids_path.read_text(encoding="utf-8"))
        ids = payload["ids"]
        if len(ids) != len(vectors):
            raise ValueError(
                f"Uyğunsuzluq: {self.ids_path.name}-də {len(ids)} id, "
                f"{self.emb_path.name}-də {len(vectors)} vektor var."
            )
        self._vectors = validate_vectors(vectors)
        self._ids = list(ids)
        self._meta = dict(payload.get("meta", {}))
        return True

    def save(self) -> None:
        """`.npy` + `ids.json` yazır (atomik: əvvəlcə .tmp, sonra replace)."""
        self.emb_path.parent.mkdir(parents=True, exist_ok=True)
        self.ids_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_emb = self.emb_path.with_suffix(self.emb_path.suffix + ".tmp")
        # Fayl deskriptoruna yazırıq: np.save(path, ...) yolun sonuna ".npy"
        # əlavə edir və .tmp faylı gözlənilən adla yaranmır.
        with open(tmp_emb, "wb") as handle:
            np.save(handle, self._vectors)
        tmp_emb.replace(self.emb_path)

        payload = {
            "model_id": config.MODEL_ID,
            "model_ver": config.MODEL_VER,
            "dim": config.EMB_DIM,
            "dtype": config.EMB_DTYPE,
            "normalized": True,
            "count": len(self._ids),
            "ids": self._ids,
            "meta": self._meta,
        }
        tmp_ids = self.ids_path.with_suffix(self.ids_path.suffix + ".tmp")
        tmp_ids.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp_ids.replace(self.ids_path)

    # --- VectorStore ------------------------------------------------------
    def add(
        self,
        ids: Sequence[str],
        vecs: np.ndarray,
        meta: Sequence[dict[str, Any]] | None = None,
    ) -> int:
        ids = [str(i) for i in ids]
        vecs = validate_vectors(vecs)
        if len(ids) != len(vecs):
            raise ValueError(f"{len(ids)} id, {len(vecs)} vektor — sayları uyğun deyil")
        meta = list(meta) if meta is not None else [{} for _ in ids]
        if len(meta) != len(ids):
            raise ValueError(f"{len(ids)} id, {len(meta)} meta — sayları uyğun deyil")

        index = {item_id: pos for pos, item_id in enumerate(self._ids)}
        new_ids, new_rows = [], []
        for item_id, vec, item_meta in zip(ids, vecs, meta):
            self._meta[item_id] = dict(item_meta)
            if item_id in index:  # upsert
                self._vectors[index[item_id]] = vec
            else:
                new_ids.append(item_id)
                new_rows.append(vec)
        if new_rows:
            self._vectors = np.concatenate(
                [self._vectors, np.stack(new_rows).astype(np.float32)], axis=0
            )
            self._ids.extend(new_ids)
        return len(ids)

    def search(
        self,
        vec: np.ndarray,
        k: int = config.DEFAULT_TOP_K,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not self._ids:
            return []
        query = as_query_vector(vec)
        # Vektorlar L2-normalized olduğu üçün dot product = cosine similarity.
        scores = self._vectors @ query

        candidates = range(len(self._ids))
        if where:
            candidates = [
                i for i in candidates
                if matches_filter(self._meta.get(self._ids[i], {}), where)
            ]
            if not candidates:
                return []
        candidates = np.asarray(list(candidates), dtype=np.int64)

        k = max(0, min(k, len(candidates)))
        if k == 0:
            return []
        subset = scores[candidates]
        # argpartition: tam sıralamadan ucuz, sonra yalnız top-k sıralanır.
        top = candidates[np.argpartition(-subset, k - 1)[:k]]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [
            SearchResult(
                item_id=self._ids[i],
                score=float(scores[i]),
                meta=dict(self._meta.get(self._ids[i], {})),
            )
            for i in top
        ]

    def count(self) -> int:
        return len(self._ids)

    def get(self, item_id: str) -> SearchResult | None:
        if item_id not in self._meta:
            return None
        return SearchResult(item_id=item_id, score=1.0, meta=dict(self._meta[item_id]))

    # --- əlavə (numpy-a xas) ---------------------------------------------
    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

    def distinct_values(self, field: str) -> list[str]:
        return sorted({
            str(m[field]) for m in self._meta.values()
            if m.get(field) is not None
        })

    def vector_of(self, item_id: str) -> np.ndarray | None:
        try:
            return self._vectors[self._ids.index(item_id)]
        except ValueError:
            return None


def main() -> int:
    parser = argparse.ArgumentParser(description="numpy store məlumatı")
    parser.add_argument("--emb-path", default=config.EMB_PATH)
    parser.add_argument("--ids-path", default=config.IDS_PATH)
    parser.add_argument("--head", type=int, default=5, help="neçə id göstərilsin")
    args = parser.parse_args()

    store = NumpyStore(args.emb_path, args.ids_path)
    print(f"emb_path : {store.emb_path}")
    print(f"ids_path : {store.ids_path}")
    print(f"count    : {store.count()}")
    if store.count():
        norms = np.linalg.norm(store.vectors, axis=1)
        print(f"shape    : {store.vectors.shape} dtype={store.vectors.dtype}")
        print(f"norm     : min={norms.min():.6f} max={norms.max():.6f}")
        for item_id in store.ids[: args.head]:
            print(f"  {item_id}  {store._meta.get(item_id, {})}")
    else:
        print("(boş — əvvəlcə ingest.py işlət)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
