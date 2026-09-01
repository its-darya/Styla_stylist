"""Outfit Style Dataset və Pseudo-Labeling Alətləri.

D-nin hazırladığı pseudo-label-lənmiş tam outfit-lər (top+bottom embedding-ləri
və stil etiketləri) üçün PyTorch Dataset və köməkçi funksiyalar.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ml.style import config


class StyleEncoder:
    """Stil kateqoriyası adları (string) və rəqəm indeksləri (int) arasında çevirici."""

    def __init__(self, class_names: Optional[Sequence[str]] = None) -> None:
        self.classes = list(class_names or config.STYLES)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.idx_to_class = {i: name for i, name in enumerate(self.classes)}

    def encode(self, name: str) -> int:
        if name not in self.class_to_idx:
            raise ValueError(f"Naməlum stil adı: {name!r}. İcazəli: {self.classes}")
        return self.class_to_idx[name]

    def decode(self, idx: int) -> str:
        return self.idx_to_class[idx]

    def num_classes(self) -> int:
        return len(self.classes)

    def save(self, file_path: Union[str, Path]) -> None:
        Path(file_path).write_text(
            json.dumps(self.classes, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> StyleEncoder:
        classes = json.loads(Path(file_path).read_text(encoding="utf-8"))
        return cls(classes)


class OutfitStyleDataset(Dataset):
    """Top + Bottom embedding-ləri və stil etiketləri üçün PyTorch Dataset."""

    def __init__(
        self,
        top_embeddings: np.ndarray | torch.Tensor,
        bottom_embeddings: np.ndarray | torch.Tensor,
        labels: Sequence[Union[int, str]] | np.ndarray | torch.Tensor,
        encoder: Optional[StyleEncoder] = None,
    ) -> None:
        self.encoder = encoder or StyleEncoder()

        if len(top_embeddings) != len(bottom_embeddings) or len(top_embeddings) != len(labels):
            raise ValueError(
                f"Ölçü fərqi: top={len(top_embeddings)}, bottom={len(bottom_embeddings)}, labels={len(labels)}"
            )

        self.top_embs = torch.as_tensor(top_embeddings, dtype=torch.float32)
        self.bottom_embs = torch.as_tensor(bottom_embeddings, dtype=torch.float32)

        # Etiketləri int indekslərə çeviririk
        if isinstance(labels[0], str):
            int_labels = [self.encoder.encode(lbl) for lbl in labels]
        else:
            int_labels = [int(lbl) for lbl in labels]

        self.labels = torch.as_tensor(int_labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.top_embs[idx], self.bottom_embs[idx], self.labels[idx]


def load_style_data_from_npz(
    file_path: Union[str, Path]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """D-dən gələn stil dataset faylını (.npz) oxuyur."""
    data = np.load(file_path, allow_pickle=True)
    top_embs = np.asarray(data["top_embeddings"], dtype=np.float32)
    bottom_embs = np.asarray(data["bottom_embeddings"], dtype=np.float32)
    labels = np.asarray(data["labels"])
    classes = list(data["classes"]) if "classes" in data else config.STYLES
    return top_embs, bottom_embs, labels, classes


def save_style_data_to_npz(
    top_embs: np.ndarray,
    bottom_embs: np.ndarray,
    labels: Union[np.ndarray, Sequence[str]],
    file_path: Union[str, Path],
    classes: Optional[Sequence[str]] = None,
) -> Path:
    """Stil datasetini .npz faylında saxlayır."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    classes = list(classes or config.STYLES)
    np.savez_compressed(
        file_path,
        top_embeddings=top_embs,
        bottom_embeddings=bottom_embs,
        labels=labels,
        classes=classes,
    )
    return file_path


def generate_pseudo_labels_with_clip(
    outfits: list[dict[str, Any]],
    embedder: Any,
    styles: Sequence[str] = config.STYLES,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """D üçün köməkçi: CLIP zero-shot text classifier ilə outfit-lərə pseudo-label verir.

    Hər outfit üçün:
        1. Top və Bottom əşyalarının embedding-ləri çıxarılır
        2. Ortaq vizual vektor CLIP stil promptları ilə tutuşdurulur
        3. Ən yüksək oxşarlığa malik stil adı pseudo-label təyin edilir.
    """
    style_prompts = [config.STYLE_PROMPT_TEMPLATE.format(s) for s in styles]
    text_vectors = embedder.embed_texts(style_prompts)  # [num_styles, 512]

    top_list = []
    bottom_list = []
    pseudo_labels = []

    for outfit in outfits:
        top_path = outfit.get("top_path")
        bottom_path = outfit.get("bottom_path")
        if not top_path or not bottom_path:
            continue

        top_emb = embedder.embed_images([top_path])[0]
        bottom_emb = embedder.embed_images([bottom_path])[0]

        # Ortaq outfit vektoru
        outfit_vec = (top_emb + bottom_emb) / 2.0
        outfit_vec = outfit_vec / np.linalg.norm(outfit_vec)

        # Kosinus oxşarlığı ilə ən yaxın stili tapırıq
        sims = np.dot(text_vectors, outfit_vec)
        best_idx = int(np.argmax(sims))
        best_style = styles[best_idx]

        top_list.append(top_emb)
        bottom_list.append(bottom_emb)
        pseudo_labels.append(best_style)

    return (
        np.asarray(top_list, dtype=np.float32),
        np.asarray(bottom_list, dtype=np.float32),
        pseudo_labels,
    )


def split_style_dataset(
    dataset: OutfitStyleDataset,
    val_split: float = config.VAL_SPLIT,
    test_split: float = config.TEST_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> tuple[OutfitStyleDataset, OutfitStyleDataset, OutfitStyleDataset]:
    """OutfitStyleDataset obyektini Train, Val və Test hissələrinə bölür."""
    n = len(dataset)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)

    test_size = int(n * test_split)
    val_size = int(n * val_split)
    train_size = n - val_size - test_size

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    encoder = dataset.encoder

    train_ds = OutfitStyleDataset(
        dataset.top_embs[train_idx],
        dataset.bottom_embs[train_idx],
        dataset.labels[train_idx],
        encoder=encoder,
    )
    val_ds = OutfitStyleDataset(
        dataset.top_embs[val_idx],
        dataset.bottom_embs[val_idx],
        dataset.labels[val_idx],
        encoder=encoder,
    )
    test_ds = OutfitStyleDataset(
        dataset.top_embs[test_idx],
        dataset.bottom_embs[test_idx],
        dataset.labels[test_idx],
        encoder=encoder,
    )

    return train_ds, val_ds, test_ds
