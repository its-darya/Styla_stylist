"""FashionCLIP çəkilərini yerli kеşə endirir (offline işləmək üçün).

İstifadə:
    python -m ml.retrieval.scripts.download_model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from huggingface_hub import snapshot_download

from ml.retrieval import config

# Yalnız PyTorch/safetensors variantı — TF/Flax çəkiləri lazım deyil.
ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.safetensors",
    "*.model",
]
IGNORE_PATTERNS = [
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "flax_model*",
    "tf_model*",
    "pytorch_model.bin",  # safetensors kifayətdir
]


def download(model_id: str = config.MODEL_ID, cache_dir: Path | None = None) -> Path:
    cache_dir = Path(cache_dir or config.MODEL_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir),
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )
    return Path(path)


def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description="FashionCLIP çəkilərini endir")
    parser.add_argument("--model-id", default=config.MODEL_ID)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    print(f"Endirilir: {args.model_id}")
    path = download(args.model_id, args.cache_dir)
    print(f"Yerli yol : {path}")
    print(f"Ölçü      : {_dir_size_mb(path):.1f} MB")
    for f in sorted(path.rglob("*")):
        if f.is_file():
            print(f"  {f.name:<32} {f.stat().st_size / 1024 / 1024:8.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
