"""Şəkil qovluğunu embed edib vector store-lara yazır.

Qayda: embedding-lər HƏM diskə (`.npy` + `ids.json`), HƏM DB-yə yazılır.
Disk artefaktları həmişə yazılır (A modulu onları DB-siz oxuyur); əlavə
backend-lər `--backends` və ya `config.INGEST_BACKENDS` ilə seçilir.

İstifadə:
    python -m ml.retrieval.ingest --image-dir data/images
    python -m ml.retrieval.ingest --backends numpy,pg
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.retrieval import config
from ml.retrieval.embedder import FashionCLIPEmbedder
from ml.retrieval.store.base import get_store
from ml.retrieval.store.numpy_store import NumpyStore

DISK_BACKEND = "numpy"


def discover_images(image_dir: Path) -> list[Path]:
    """Qovluqdakı şəkilləri ada görə sıralı qaytarır (deterministik sıra)."""
    return sorted(
        p for p in Path(image_dir).iterdir()
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def load_sample_meta(image_dir: Path) -> dict[str, dict[str, Any]]:
    """`meta.json` varsa oxuyur (download_sample_images.py yazır). Yoxdursa boş."""
    meta_path = Path(image_dir) / Path(config.IMAGE_META_PATH).name
    if not meta_path.exists():
        return {}
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return payload.get("items", {})


def build_meta(
    path: Path, item_id: str, sample_meta: dict[str, dict[str, Any]], source: str
) -> dict[str, Any]:
    """Bir əşyanın metadata-sı. B modulu category/color-u sonradan zənginləşdirə bilər."""
    info = sample_meta.get(item_id, {})
    return {
        "image_path": str(path),
        "category": info.get("category"),
        "color": info.get("color"),
        "text": info.get("text"),
        "outfit_id": info.get("outfit_id"),
        "source_id": info.get("source_id"),
        "model_ver": config.MODEL_VER,
        "source": source,
    }


def ingest(
    image_dir: Path = config.IMAGE_DIR,
    backends: list[str] | None = None,
    source: str = "wardrobe",
    limit: int | None = None,
    embedder: FashionCLIPEmbedder | None = None,
) -> dict[str, Any]:
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Şəkil qovluğu yoxdur: {image_dir}\n"
            "Əvvəlcə: python -m ml.retrieval.scripts.download_sample_images"
        )

    paths = discover_images(image_dir)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"{image_dir} qovluğunda şəkil tapılmadı")

    sample_meta = load_sample_meta(image_dir)
    ids = [p.stem for p in paths]
    metas = [build_meta(p, i, sample_meta, source) for p, i in zip(paths, ids)]

    embedder = embedder or FashionCLIPEmbedder()
    print(f"{len(paths)} şəkil embed edilir (batch={embedder.batch_size}, device={embedder.device})...")
    t0 = time.perf_counter()
    vectors = embedder.embed_images(paths)
    embed_sec = time.perf_counter() - t0
    print(f"  {embed_sec:.2f} san ({embed_sec / len(paths) * 1000:.0f} ms/şəkil)")

    # Disk artefaktları HƏMİŞƏ yazılır — A modulu bunları DB-siz oxuyur.
    backends = list(backends or config.INGEST_BACKENDS)
    if DISK_BACKEND not in backends:
        backends.insert(0, DISK_BACKEND)

    written: dict[str, int] = {}
    for backend in backends:
        store = get_store(backend)
        try:
            n = store.add(ids, vectors, metas)
            if isinstance(store, NumpyStore):
                store.save()
                print(f"  [{backend}] {n} vektor -> {store.emb_path.name} + {store.ids_path.name}")
            else:
                print(f"  [{backend}] {n} vektor -> {backend} store")
            written[backend] = n
        finally:
            store.close()

    return {
        "count": len(ids),
        "ids": ids,
        "vectors": vectors,
        "embed_sec": embed_sec,
        "written": written,
        "image_dir": str(image_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Şəkilləri embed edib store-a yaz")
    parser.add_argument("--image-dir", default=config.IMAGE_DIR)
    parser.add_argument("--backends", default=",".join(config.INGEST_BACKENDS),
                        help="vergüllə: numpy,pg (numpy həmişə əlavə olunur)")
    parser.add_argument("--source", default="wardrobe", help="mənbə etiketi (wardrobe/catalog)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    result = ingest(
        image_dir=Path(args.image_dir),
        backends=backends,
        source=args.source,
        limit=args.limit,
    )

    vectors = result["vectors"]
    norms = np.linalg.norm(vectors, axis=1)
    print("\n--- xülasə ---")
    print(f"əşya sayı : {result['count']}")
    print(f"shape     : {vectors.shape}  dtype={vectors.dtype}")
    print(f"L2 norm   : min={norms.min():.6f} max={norms.max():.6f}")
    print(f"embed vaxtı: {result['embed_sec']:.2f} san")
    print(f"yazıldı   : {result['written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
