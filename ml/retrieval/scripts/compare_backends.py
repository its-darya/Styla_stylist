"""İki backend-in (numpy vs pgvector) EYNİ nəticə verdiyini yoxlayır.

Hər ikisi exact cosine search etdiyi üçün top-k id sırası eyni olmalı,
score-lar isə float dəqiqliyi həddində üst-üstə düşməlidir.

Şərt: DB işləməlidir (docker compose up -d) və hər iki backend-ə ingest
edilməlidir:  python -m ml.retrieval.ingest --backends numpy,pg

İstifadə:
    python -m ml.retrieval.scripts.compare_backends
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.retrieval import config
from ml.retrieval.embedder import FashionCLIPEmbedder
from ml.retrieval.store.base import get_store

SCORE_TOLERANCE = 1e-5
TEXT_QUERIES = [
    "black leather boots",
    "a floral summer dress",
    "blue denim jeans",
    "a brown handbag",
    "sunglasses",
]
_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


def compare_one(label: str, vec: np.ndarray, np_store, pg_store, k: int, where=None) -> None:
    a = np_store.search(vec, k=k, where=where)
    b = pg_store.search(vec, k=k, where=where)

    ids_a = [r.item_id for r in a]
    ids_b = [r.item_id for r in b]
    check(f"{label}: top-{k} id sırası eyni", ids_a == ids_b,
          "" if ids_a == ids_b else f"\n        numpy={ids_a}\n        pg   ={ids_b}")

    if ids_a == ids_b and a:
        diffs = [abs(x.score - y.score) for x, y in zip(a, b)]
        check(f"{label}: score fərqi < {SCORE_TOLERANCE:g}",
              max(diffs) < SCORE_TOLERANCE, f"maks fərq={max(diffs):.2e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="numpy vs pgvector müqayisəsi")
    parser.add_argument("-k", "--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--db-url", default=config.DB_URL)
    args = parser.parse_args()

    print("=" * 66)
    print("Styla · backend müqayisəsi: numpy vs pgvector (exact cosine)")
    print("=" * 66)

    np_store = get_store("numpy")
    pg_store = get_store("pg", db_url=args.db_url)

    print("\n[store vəziyyəti]")
    n_np, n_pg = np_store.count(), pg_store.count()
    print(f"  numpy: {n_np} əşya | pgvector: {n_pg} əşya")
    check("hər iki store-da eyni sayda əşya", n_np == n_pg, f"{n_np} vs {n_pg}")
    check("pgvector-da ANN index yoxdur (exact search)",
          not pg_store.ann_indexes(),
          f"indekslər={pg_store.indexes()} ann={pg_store.ann_indexes()}")
    if n_np == 0 or n_pg == 0:
        print("\nStore boşdur -> python -m ml.retrieval.ingest --backends numpy,pg")
        return 1

    embedder = FashionCLIPEmbedder()

    print("\n[mətn sorğuları]")
    for q in TEXT_QUERIES:
        compare_one(f'"{q}"', embedder.embed_texts([q])[0], np_store, pg_store, args.top_k)

    print("\n[şəkil sorğuları]")
    image_ids = np_store.ids[:3]
    for item_id in image_ids:
        path = np_store._meta[item_id]["image_path"]
        compare_one(f"şəkil {item_id}", embedder.embed_images([path])[0],
                    np_store, pg_store, args.top_k)

    print("\n[kateqoriya filtri]")
    categories = [m.get("category") for m in np_store._meta.values() if m.get("category")]
    if categories:
        top_cat = max(set(categories), key=categories.count)
        compare_one(f'filtr category="{top_cat}"',
                    embedder.embed_texts(["clothing"])[0],
                    np_store, pg_store, args.top_k, where={"category": top_cat})

    print("\n[bütün əşyalar üzrə tam sıralama]")
    mismatched = 0
    for item_id in np_store.ids:
        vec = np_store.vector_of(item_id)
        ia = [r.item_id for r in np_store.search(vec, k=args.top_k)]
        ib = [r.item_id for r in pg_store.search(vec, k=args.top_k)]
        if ia != ib:
            mismatched += 1
            if mismatched <= 3:
                print(f"        fərq {item_id}: numpy={ia} pg={ib}")
    check(f"{len(np_store.ids)} sorğunun hamısında top-{args.top_k} eyni",
          mismatched == 0, f"{mismatched} fərq")

    np_store.close()
    pg_store.close()

    print("\n" + "=" * 66)
    if _failures:
        print(f"NƏTİCƏ: {len(_failures)} yoxlama uğursuz -> {_failures}")
        return 1
    print("NƏTİCƏ: iki backend eyni nəticə verir ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
