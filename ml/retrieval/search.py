"""Retrieval sorğusu — şəkil VƏ YA mətn.

Hər iki modallıq eyni [EMB_DIM] vektor fəzasına düşür (CLIP), ona görə
axtarış kodu eynidir: sorğunu embed et -> store.search().

İstifadə:
    python -m ml.retrieval.search --text "black leather boots" -k 5
    python -m ml.retrieval.search --image data/images/item_0002.jpg -k 5
    python -m ml.retrieval.search --text "dress" --category "Day Dresses"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.retrieval import config
from ml.retrieval.embedder import FashionCLIPEmbedder
from ml.retrieval.store.base import SearchResult, VectorStore, get_store


class Searcher:
    """Embedder + store cütü. Model və store bir dəfə yüklənir, təkrar istifadə olunur."""

    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: FashionCLIPEmbedder | None = None,
        backend: str | None = None,
    ) -> None:
        self.store = store or get_store(backend)
        self.embedder = embedder or FashionCLIPEmbedder()

    # --- sorğu vektoru ----------------------------------------------------
    def encode_text(self, text: str) -> np.ndarray:
        return self.embedder.embed_texts([text])[0]

    def encode_image(self, path: str | Path) -> np.ndarray:
        return self.embedder.embed_images([path])[0]

    def encode(self, *, text: str | None = None, image: str | Path | None = None) -> np.ndarray:
        if (text is None) == (image is None):
            raise ValueError("Dəqiq bir sorğu ver: ya --text, ya --image")
        return self.encode_text(text) if text is not None else self.encode_image(image)

    # --- axtarış ----------------------------------------------------------
    def search(
        self,
        *,
        text: str | None = None,
        image: str | Path | None = None,
        vector: np.ndarray | None = None,
        k: int = config.DEFAULT_TOP_K,
        where: dict[str, Any] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        """top-k nəticə. `exclude` — nəticədən çıxarılacaq id-lər (məs. sorğunun özü)."""
        if vector is None:
            vector = self.encode(text=text, image=image)
        exclude = set(exclude or ())
        # Çıxarılanlar top-k-nı doldura bilməsin deyə bir qədər artıq götürürük.
        raw = self.store.search(vector, k=k + len(exclude), where=where)
        results = [r for r in raw if r.item_id not in exclude]
        return results[:k]

    def search_timed(self, **kwargs) -> tuple[list[SearchResult], dict[str, float]]:
        """Nəticə + latency (embed və search ayrı-ayrı, saniyə)."""
        text, image = kwargs.get("text"), kwargs.get("image")
        t0 = time.perf_counter()
        vector = kwargs.pop("vector", None)
        if vector is None:
            vector = self.encode(text=text, image=image)
        embed_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        kwargs.pop("text", None)
        kwargs.pop("image", None)
        results = self.search(vector=vector, **kwargs)
        search_sec = time.perf_counter() - t1
        return results, {
            "embed_sec": embed_sec,
            "search_sec": search_sec,
            "total_sec": embed_sec + search_sec,
        }

    def close(self) -> None:
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def format_results(results: list[SearchResult]) -> str:
    if not results:
        return "  (nəticə yoxdur)"
    lines = []
    for rank, r in enumerate(results, start=1):
        label = r.meta.get("text") or r.meta.get("category") or ""
        cat = r.meta.get("category") or "-"
        lines.append(f"  {rank}. {r.item_id}  cos={r.score:.4f}  [{cat}]  {label}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Şəkil və ya mətn ilə axtarış")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="mətn sorğusu")
    group.add_argument("--image", help="şəkil sorğusu (fayl yolu)")
    parser.add_argument("-k", "--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--backend", default=None, help="numpy | pg (default: config.BACKEND)")
    parser.add_argument("--category", default=None, help="kateqoriya filtri")
    parser.add_argument("--exclude-self", action="store_true",
                        help="şəkil sorğusunda sorğunun öz id-sini nəticədən çıxar")
    args = parser.parse_args()

    where = {"category": args.category} if args.category else None
    exclude = [Path(args.image).stem] if (args.image and args.exclude_self) else None

    with Searcher(backend=args.backend) as searcher:
        if searcher.store.count() == 0:
            print("Store boşdur. Əvvəlcə: python -m ml.retrieval.ingest")
            return 1
        results, timing = searcher.search_timed(
            text=args.text, image=args.image, k=args.top_k, where=where, exclude=exclude
        )
        query_desc = f'mətn "{args.text}"' if args.text else f"şəkil {args.image}"
        print(f"Sorğu    : {query_desc}")
        print(f"Backend  : {args.backend or config.BACKEND}  ({searcher.store.count()} əşya)")
        if where:
            print(f"Filtr    : {where}")
        print(f"Top-{args.top_k}:")
        print(format_results(results))
        print(f"\nLatency  : embed={timing['embed_sec'] * 1000:.0f} ms  "
              f"search={timing['search_sec'] * 1000:.1f} ms  "
              f"cəmi={timing['total_sec'] * 1000:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
