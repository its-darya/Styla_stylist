"""Embedder smoke test — real şəkil tələb etmir (sintetik şəkillər yaradır).

Yoxlanılanlar:
    1. embed_images  -> shape [3, 512], dtype float32, ‖v‖₂ ≈ 1.0
    2. embed_texts   -> shape [3, 512], dtype float32, ‖v‖₂ ≈ 1.0
    3. Determinizm   -> eyni giriş, eyni çıxış
    4. Cosine sağlamlığı -> özü ilə oxşarlıq = 1.0, cross-modal [-1, 1] aralığında
    5. CPU rejimi    -> torch CUDA build deyil, thread sayı config-dən

İstifadə:
    python -m ml.retrieval.scripts.smoke_test
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.retrieval import config
from ml.retrieval.embedder import FashionCLIPEmbedder

NORM_TOLERANCE = 1e-4
SMOKE_IMAGE_SIZE = (224, 224)
SMOKE_COLORS = [(220, 40, 40), (40, 120, 220), (30, 30, 30)]
SMOKE_TEXTS = ["a red t-shirt", "blue denim jeans", "black leather boots"]

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


def make_synthetic_images(directory: Path) -> list[Path]:
    """Sadə rəngli kvadratlar — modelin borusunu yoxlamaq üçün kifayətdir."""
    paths = []
    for i, color in enumerate(SMOKE_COLORS, start=1):
        path = directory / f"smoke_{i}.png"
        Image.new("RGB", SMOKE_IMAGE_SIZE, color).save(path)
        paths.append(path)
    return paths


def check_contract(name: str, vectors: np.ndarray, expected_n: int) -> None:
    check(f"{name}: shape == [{expected_n}, {config.EMB_DIM}]",
          vectors.shape == (expected_n, config.EMB_DIM), str(vectors.shape))
    check(f"{name}: dtype == float32",
          vectors.dtype == np.float32, str(vectors.dtype))
    norms = np.linalg.norm(vectors, axis=1)
    check(f"{name}: L2 norm ≈ 1.0",
          bool(np.allclose(norms, 1.0, atol=NORM_TOLERANCE)),
          f"norms={norms.round(6).tolist()}")
    check(f"{name}: NaN/Inf yoxdur", bool(np.isfinite(vectors).all()))


def main() -> int:
    print("=" * 62)
    print("Styla · ml/retrieval · embedder smoke test")
    print("=" * 62)

    print("\n[mühit]")
    check("torch CPU build (cuda_available == False)",
          not torch.cuda.is_available(),
          f"torch={torch.__version__} cuda_build={torch.version.cuda}")
    check("config.DEVICE == 'cpu'", config.DEVICE == "cpu", config.DEVICE)

    embedder = FashionCLIPEmbedder()
    check("torch.set_num_threads tətbiq olundu",
          torch.get_num_threads() == config.NUM_THREADS,
          f"{torch.get_num_threads()} == config.NUM_THREADS({config.NUM_THREADS})")
    check("batch_size config-dən",
          embedder.batch_size == config.BATCH_SIZE, str(embedder.batch_size))

    with tempfile.TemporaryDirectory() as tmp:
        paths = make_synthetic_images(Path(tmp))

        print("\n[şəkil embedding]")
        t0 = time.perf_counter()
        img_vecs = embedder.embed_images(paths)
        img_sec = time.perf_counter() - t0
        check_contract("images", img_vecs, len(paths))
        print(f"        {len(paths)} şəkil / {img_sec:.2f} san "
              f"({img_sec / len(paths):.2f} san per şəkil)")

        print("\n[mətn embedding]")
        t0 = time.perf_counter()
        txt_vecs = embedder.embed_texts(SMOKE_TEXTS)
        txt_sec = time.perf_counter() - t0
        check_contract("texts", txt_vecs, len(SMOKE_TEXTS))
        print(f"        {len(SMOKE_TEXTS)} mətn / {txt_sec:.2f} san")

        print("\n[determinizm]")
        repeat = embedder.embed_images(paths)
        check("eyni şəkil -> eyni vektor",
              bool(np.allclose(img_vecs, repeat, atol=1e-6)),
              f"maks fərq={np.abs(img_vecs - repeat).max():.2e}")

        print("\n[cosine sağlamlığı]")
        self_sim = float(img_vecs[0] @ img_vecs[0])
        check("özü ilə cosine == 1.0", abs(self_sim - 1.0) < 1e-4, f"{self_sim:.6f}")
        cross = img_vecs @ txt_vecs.T
        check("cross-modal oxşarlıq [-1, 1] aralığında",
              bool((cross >= -1.0 - 1e-4).all() and (cross <= 1.0 + 1e-4).all()),
              f"min={cross.min():.4f} max={cross.max():.4f}")

        print("\n[boş giriş]")
        check("embed_images([]) -> [0, 512]",
              embedder.embed_images([]).shape == (0, config.EMB_DIM))
        check("embed_texts([]) -> [0, 512]",
              embedder.embed_texts([]).shape == (0, config.EMB_DIM))

    print("\n" + "=" * 62)
    if _failures:
        print(f"NƏTİCƏ: {len(_failures)} test uğursuz -> {_failures}")
        return 1
    print("NƏTİCƏ: bütün testlər uğurlu ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
