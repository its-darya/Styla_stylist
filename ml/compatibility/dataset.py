"""Compatibility Dataset və Data Loader-lər.

D-nin Polyvore-dan hazırladığı pozitiv (real outfit-lərdəki cütlər) və
neqativ (təsadüfi yığılmış cütlər) nümunələrini və hesablanmış
FashionCLIP embedding-lərini təlim və qiymətləndirmə üçün hazırlayır.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ml.compatibility import config


class PairCompatibilityDataset(Dataset):
    """Embedding cütləri və binar etiketlər (1: uyğun, 0: uyğunsuz) üçün PyTorch Dataset."""

    def __init__(
        self,
        embeddings1: np.ndarray | torch.Tensor,
        embeddings2: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
    ) -> None:
        """
        Args:
            embeddings1: [N, 512] ölçülü 1-ci əşyaların embedding-ləri
            embeddings2: [N, 512] ölçülü 2-ci əşyaların embedding-ləri
            labels: [N] ölçülü etiketlər (1.0 = pozitiv cüt, 0.0 = neqativ cüt)
        """
        if len(embeddings1) != len(embeddings2) or len(embeddings1) != len(labels):
            raise ValueError(
                f"Ölçülər uyğunsuz: e1={len(embeddings1)}, e2={len(embeddings2)}, labels={len(labels)}"
            )

        self.e1 = torch.as_tensor(embeddings1, dtype=torch.float32)
        self.e2 = torch.as_tensor(embeddings2, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.e1[idx], self.e2[idx], self.labels[idx]


def load_pairs_from_npz(
    npz_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D-nin saxladığı .npz faylından e1, e2 və labels massivlərini oxuyur."""
    data = np.load(npz_path, allow_pickle=True)
    e1 = data["embeddings1"] if "embeddings1" in data else data["e1"]
    e2 = data["embeddings2"] if "embeddings2" in data else data["e2"]
    labels = data["labels"] if "labels" in data else data["y"]
    return np.asarray(e1, dtype=np.float32), np.asarray(e2, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def save_pairs_to_npz(
    e1: np.ndarray,
    e2: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path,
) -> Path:
    """Cütləri və etiketləri .npz faylına yazır."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, embeddings1=e1, embeddings2=e2, labels=labels)
    return save_path


def create_pairs_from_outfits(
    item_embeddings: dict[str, np.ndarray],
    outfit_to_items: dict[str, list[str]],
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Polyvore outfit metadata və embedding lüğətindən pozitiv və neqativ cütlər yaradır.

    - Pozitiv: eyni outfit içindəki bütün kombinasiyalar (item_i, item_j).
    - Neqativ: fərqli outfit-lərdən təsadüfi seçilmiş əşya cütləri.
    """
    rng = random.Random(seed)
    
    # 1. Pozitiv cütlərin yığılması
    pos_pairs: list[tuple[str, str]] = []
    for outfit_id, items in outfit_to_items.items():
        # Yalnız embedding-i mövcud olan əşyaları saxlayırıq
        valid_items = [item for item in items if item in item_embeddings]
        n = len(valid_items)
        for i in range(n):
            for j in range(i + 1, n):
                pos_pairs.append((valid_items[i], valid_items[j]))

    if not pos_pairs:
        raise ValueError("Heç bir pozitiv cüt tapılmadı! Lütfən outfit və embedding məlumatlarını yoxlayın.")

    num_pos = len(pos_pairs)
    num_neg = int(num_pos * negative_ratio)

    # 2. Neqativ cütlərin yığılması (Hard/Random negative sampling)
    # Hər əşyanın aid olduğu outfit_id-ni xəritələyirik
    item_to_outfit: dict[str, str] = {}
    for outfit_id, items in outfit_to_items.items():
        for item in items:
            item_to_outfit[item] = outfit_id

    all_items = [i for i in item_embeddings.keys() if i in item_to_outfit]
    if len(all_items) < 2:
        raise ValueError("Neqativ cüt yaratmaq üçün ən azı 2 fərqli əşya lazımdır.")

    neg_pairs: list[tuple[str, str]] = []
    attempts = 0
    max_attempts = num_neg * 20

    while len(neg_pairs) < num_neg and attempts < max_attempts:
        attempts += 1
        item_a = rng.choice(all_items)
        item_b = rng.choice(all_items)
        if item_a == item_b:
            continue
        # Fərqli outfit-lərdən olmalıdır
        if item_to_outfit[item_a] != item_to_outfit[item_b]:
            neg_pairs.append((item_a, item_b))

    # Cütləri vektorlara çeviririk
    e1_list: list[np.ndarray] = []
    e2_list: list[np.ndarray] = []
    labels_list: list[float] = []

    for item_a, item_b in pos_pairs:
        e1_list.append(item_embeddings[item_a])
        e2_list.append(item_embeddings[item_b])
        labels_list.append(1.0)

    for item_a, item_b in neg_pairs:
        e1_list.append(item_embeddings[item_a])
        e2_list.append(item_embeddings[item_b])
        labels_list.append(0.0)

    e1 = np.asarray(e1_list, dtype=np.float32)
    e2 = np.asarray(e2_list, dtype=np.float32)
    labels = np.asarray(labels_list, dtype=np.float32)

    # Cütləri qarışdırırıq (shuffle)
    indices = np.arange(len(labels))
    rng.shuffle(indices)

    return e1[indices], e2[indices], labels[indices]


def split_dataset(
    e1: np.ndarray,
    e2: np.ndarray,
    labels: np.ndarray,
    val_split: float = config.VAL_SPLIT,
    test_split: float = config.TEST_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> tuple[
    PairCompatibilityDataset,
    PairCompatibilityDataset,
    PairCompatibilityDataset,
]:
    """Dataseti Train, Val və Test hissələrinə bölür."""
    num_samples = len(labels)
    indices = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)

    test_size = int(num_samples * test_split)
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size - test_size

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    train_ds = PairCompatibilityDataset(e1[train_idx], e2[train_idx], labels[train_idx])
    val_ds = PairCompatibilityDataset(e1[val_idx], e2[val_idx], labels[val_idx])
    test_ds = PairCompatibilityDataset(e1[test_idx], e2[test_idx], labels[test_idx])

    return train_ds, val_ds, test_ds


def get_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """DataLoader obyektlərini hazırlayır."""
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader
