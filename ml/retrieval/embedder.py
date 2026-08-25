"""FashionCLIP embedder — CPU-only.

Müqavilə (bütün modul boyu sabitdir):
    - dtype   : float32
    - shape   : [N, config.EMB_DIM]  (= [N, 512])
    - norm    : L2-normalized, ‖v‖₂ ≈ 1.0
Normalized olduğu üçün cosine similarity sadəcə dot product-dır.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image

if __package__ in (None, ""):  # birbaşa `python embedder.py` işlədiləndə
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.retrieval import config


def _configure_torch() -> None:
    """CPU-only rejimi: thread sayı config-dən, qradiyent yoxdur."""
    torch.set_num_threads(config.NUM_THREADS)
    torch.set_grad_enabled(False)


def _batched(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _features_to_tensor(output) -> torch.Tensor:
    """`get_image_features`/`get_text_features` çıxışını tensora çevirir.

    transformers 4.x düz Tensor qaytarır; 5.x isə `BaseModelOutputWithPooling`
    qaytarır və proyeksiya olunmuş [N, EMB_DIM] vektor `pooler_output`-dadır.
    Hər iki versiyanı dəstəkləyirik.
    """
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if pooled is None:
        raise TypeError(
            f"Gözlənilməz feature çıxışı: {type(output)!r} (pooler_output yoxdur)"
        )
    return pooled


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Sətir-sətir L2 normalizasiya. Sıfır vektorlar olduğu kimi qalır."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


class FashionCLIPEmbedder:
    """FashionCLIP ilə şəkil və mətn embedding-ləri.

    Model ilk `embed_*` çağırışında lazy yüklənir (import ucuz qalsın deyə).
    """

    def __init__(
        self,
        model_id: str = config.MODEL_ID,
        device: str = config.DEVICE,
        batch_size: int = config.BATCH_SIZE,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = str(cache_dir or config.MODEL_CACHE_DIR)
        self._model = None
        self._processor = None
        _configure_torch()

    # --- lazy loading -----------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, CLIPModel

        self._model = CLIPModel.from_pretrained(
            self.model_id, cache_dir=self.cache_dir
        ).to(self.device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, cache_dir=self.cache_dir
        )
        actual_dim = int(self._model.config.projection_dim)
        if actual_dim != config.EMB_DIM:
            raise ValueError(
                f"Model projection_dim={actual_dim}, config.EMB_DIM={config.EMB_DIM}. "
                "config.EMB_DIM-i və pgvector VECTOR(n) ölçüsünü uyğunlaşdır."
            )

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    @property
    def processor(self):
        self._ensure_loaded()
        return self._processor

    # --- public API -------------------------------------------------------
    def embed_images(self, paths: Sequence[str | Path]) -> np.ndarray:
        """Şəkil yollarını [N, EMB_DIM] float32 L2-normalized massivə çevirir."""
        paths = list(paths)
        if not paths:
            return np.zeros((0, config.EMB_DIM), dtype=np.float32)
        self._ensure_loaded()

        chunks = []
        for batch in _batched(paths, self.batch_size):
            images = [self._load_image(p) for p in batch]
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                feats = self._model.get_image_features(**inputs)
            chunks.append(_features_to_tensor(feats).detach().cpu().numpy())
            for img in images:
                img.close()
        return l2_normalize(np.concatenate(chunks, axis=0))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Mətnləri [N, EMB_DIM] float32 L2-normalized massivə çevirir."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, config.EMB_DIM), dtype=np.float32)
        self._ensure_loaded()

        chunks = []
        for batch in _batched(texts, self.batch_size):
            inputs = self._processor(
                text=list(batch), return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                feats = self._model.get_text_features(**inputs)
            chunks.append(_features_to_tensor(feats).detach().cpu().numpy())
        return l2_normalize(np.concatenate(chunks, axis=0))

    def embed_pil_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Artıq açılmış PIL şəkilləri üçün (outfit parsing crop-ları və s.)."""
        images = list(images)
        if not images:
            return np.zeros((0, config.EMB_DIM), dtype=np.float32)
        self._ensure_loaded()

        chunks = []
        for batch in _batched(images, self.batch_size):
            prepared = [self._composite_over_white(im) for im in batch]
            inputs = self._processor(images=prepared, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                feats = self._model.get_image_features(**inputs)
            chunks.append(_features_to_tensor(feats).detach().cpu().numpy())
        return l2_normalize(np.concatenate(chunks, axis=0))

    # --- helpers ----------------------------------------------------------
    @staticmethod
    def _composite_over_white(img: Image.Image) -> Image.Image:
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGBA')
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            return bg
        return img.convert("RGB")

    @staticmethod
    def _load_image(path: str | Path) -> Image.Image:
        img = Image.open(path)
        return FashionCLIPEmbedder._composite_over_white(img)


def main() -> int:
    parser = argparse.ArgumentParser(description="FashionCLIP embedder (CPU)")
    parser.add_argument("--image", action="append", default=[], help="şəkil yolu (təkrarlana bilər)")
    parser.add_argument("--text", action="append", default=[], help="mətn (təkrarlana bilər)")
    args = parser.parse_args()

    if not args.image and not args.text:
        parser.error("ən azı bir --image və ya --text ver")

    emb = FashionCLIPEmbedder()
    if args.image:
        vecs = emb.embed_images(args.image)
        print(f"images: shape={vecs.shape} dtype={vecs.dtype} "
              f"norms={np.linalg.norm(vecs, axis=1).round(4).tolist()}")
    if args.text:
        vecs = emb.embed_texts(args.text)
        print(f"texts : shape={vecs.shape} dtype={vecs.dtype} "
              f"norms={np.linalg.norm(vecs, axis=1).round(4).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
