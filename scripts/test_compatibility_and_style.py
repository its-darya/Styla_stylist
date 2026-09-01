"""Styla ML Modulları üçün Vahid Test və Doğrulama Skripti.

1. Compatibility Modelinin təlimi və qiymətləndirilməsi
2. Compatibility Scoring funksiyasının (0-1 arası) yoxlanılması
3. Ensemble Style Classifier modelinin təlimi və inference yoxlanışı
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Layihə kökünü sys.path-a əlavə edirik
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch

from ml.compatibility import (
    CompatibilityMLP,
    CompatibilityScorer,
    get_scorer,
    score_compatibility,
    score_compatibility_batch,
    train_compatibility_model,
)
from ml.compatibility.dataset import (
    PairCompatibilityDataset,
    get_dataloaders,
    split_dataset,
)
from ml.style import (
    EnsembleStyleMLP,
    OutfitStyleDataset,
    StyleClassifier,
    StyleEncoder,
    predict_outfit_style,
    train_style_model,
)
from ml.style.dataset import split_style_dataset


def test_task_1_compatibility_training():
    print("=" * 60)
    print("TASK 1: Compatibility Modelinin Öyrədilməsi Testi")
    print("=" * 60)

    num_samples = 400
    emb_dim = 512
    np.random.seed(42)

    # Müsbət cütlər (yüksək uyğunluq)
    pos_e1 = np.random.randn(num_samples // 2, emb_dim).astype(np.float32)
    pos_e1 /= np.linalg.norm(pos_e1, axis=1, keepdims=True)
    pos_e2 = pos_e1 + 0.25 * np.random.randn(num_samples // 2, emb_dim).astype(np.float32)
    pos_e2 /= np.linalg.norm(pos_e2, axis=1, keepdims=True)
    pos_labels = np.ones(num_samples // 2, dtype=np.float32)

    # Mənfi cütlər (uyğunsuz)
    neg_e1 = np.random.randn(num_samples // 2, emb_dim).astype(np.float32)
    neg_e1 /= np.linalg.norm(neg_e1, axis=1, keepdims=True)
    neg_e2 = np.random.randn(num_samples // 2, emb_dim).astype(np.float32)
    neg_e2 /= np.linalg.norm(neg_e2, axis=1, keepdims=True)
    neg_labels = np.zeros(num_samples // 2, dtype=np.float32)

    e1 = np.vstack([pos_e1, neg_e1])
    e2 = np.vstack([pos_e2, neg_e2])
    labels = np.concatenate([pos_labels, neg_labels])

    perm = np.random.permutation(len(labels))
    e1, e2, labels = e1[perm], e2[perm], labels[perm]

    train_ds, val_ds, test_ds = split_dataset(e1, e2, labels, seed=42)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=32
    )

    save_path = BASE_DIR / "ml" / "compatibility" / "outputs" / "test_compatibility_mlp.pt"

    model, report = train_compatibility_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=5,
        lr=1e-3,
        hidden_dims=[256, 64],
        dropout=0.2,
        mode="symmetric_full",
        save_path=save_path,
        wandb_run=None,
    )

    test_auc = report["test_metrics"]["auc"]
    test_acc = report["test_metrics"]["accuracy"]
    print(f"✓ Task 1 Uğurlu: Test AUC={test_auc:.4f}, Test Acc={test_acc:.4f}")
    assert test_auc >= 0.70, f"AUC çox aşağıdır: {test_auc}"
    assert save_path.exists(), "Model çəki faylı yaradılmadı!"
    return save_path


def test_task_2_compatibility_scoring(model_path: Path):
    print("\n" + "=" * 60)
    print("TASK 2: Compatibility Scoring Funksiyası Testi")
    print("=" * 60)

    # İki nümunə embedding vektoru yaradırıq (pgvector-dan gələn float list və ya np array)
    np.random.seed(123)
    vec_top = np.random.randn(512).astype(np.float32)
    vec_top /= np.linalg.norm(vec_top)

    # Yüksək uyğunluqlu bottom
    vec_compatible_bottom = vec_top + 0.1 * np.random.randn(512).astype(np.float32)
    vec_compatible_bottom /= np.linalg.norm(vec_compatible_bottom)

    # Təsadüfi/uyğunsuz bottom
    vec_incompatible_bottom = np.random.randn(512).astype(np.float32)
    vec_incompatible_bottom /= np.linalg.norm(vec_incompatible_bottom)

    # 1. Tək vektor cütü üçün scoring
    score_pos = score_compatibility(vec_top, vec_compatible_bottom, model_path=model_path)
    score_neg = score_compatibility(vec_top, vec_incompatible_bottom, model_path=model_path)

    print(f"Uyğun cüt balı (Pozitiv):    {score_pos:.4f} (0-1 aralığında)")
    print(f"Təsadüfi cüt balı (Neqativ): {score_neg:.4f} (0-1 aralığında)")

    assert 0.0 <= score_pos <= 1.0, f"Bal [0, 1] aralığında deyil: {score_pos}"
    assert 0.0 <= score_neg <= 1.0, f"Bal [0, 1] aralığında deyil: {score_neg}"
    assert score_pos > score_neg, f"Pozitiv cütün balı ({score_pos}) neqativdən ({score_neg}) yüksək olmalıdır!"

    # 2. Partiya (Batch) scoring
    batch_tops = np.vstack([vec_top, vec_top])
    batch_bottoms = np.vstack([vec_compatible_bottom, vec_incompatible_bottom])
    batch_scores = score_compatibility_batch(batch_tops, batch_bottoms, model_path=model_path)

    print(f"Partiya (Batch) balları: {batch_scores}")
    assert len(batch_scores) == 2
    assert batch_scores[0] > batch_scores[1]

    # 3. Python list girişi testi (pgvector formatı)
    list_top = vec_top.tolist()
    list_bottom = vec_compatible_bottom.tolist()
    list_score = score_compatibility(list_top, list_bottom, model_path=model_path)
    print(f"List formatında giriş balı: {list_score:.4f}")
    assert isinstance(list_score, float)

    print("✓ Task 2 Uğurlu: Scoring funksiyası dəqiq və tələblərə tam uyğundur.")


def test_task_3_style_classifier():
    print("\n" + "=" * 60)
    print("TASK 3: Ensemble-Level Style Classifier Testi")
    print("=" * 60)

    encoder = StyleEncoder(["casual", "formal", "streetwear", "sporty"])
    num_classes = encoder.num_classes()
    num_samples = 400
    emb_dim = 512
    np.random.seed(42)

    # Hər sinif üçün klaster mərkəzi
    centers = np.random.randn(num_classes, emb_dim).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    top_list, bottom_list, labels = [], [], []
    for i in range(num_samples):
        cls_idx = i % num_classes
        c = centers[cls_idx]
        t = c + 0.3 * np.random.randn(emb_dim).astype(np.float32)
        b = c + 0.3 * np.random.randn(emb_dim).astype(np.float32)
        t /= np.linalg.norm(t)
        b /= np.linalg.norm(b)
        top_list.append(t)
        bottom_list.append(b)
        labels.append(cls_idx)

    perm = np.random.permutation(num_samples)
    top_arr = np.array(top_list, dtype=np.float32)[perm]
    bottom_arr = np.array(bottom_list, dtype=np.float32)[perm]
    label_arr = np.array(labels)[perm]

    dataset = OutfitStyleDataset(top_arr, bottom_arr, label_arr, encoder=encoder)
    train_ds, val_ds, test_ds = split_style_dataset(dataset, seed=42)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    save_path = BASE_DIR / "ml" / "style" / "outputs" / "test_style_classifier_mlp.pt"

    model, report = train_style_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        epochs=6,
        lr=1e-3,
        hidden_dims=[256, 128],
        dropout=0.2,
        feature_mode="full_pair",
        save_path=save_path,
        encoder=encoder,
        wandb_run=None,
    )

    test_acc = report["test_metrics"]["accuracy"]
    test_f1 = report["test_metrics"]["macro_f1"]
    print(f"✓ Style Training Uğurlu: Test Top-1 Acc={test_acc:.4f}, Macro-F1={test_f1:.4f}")

    # Inference testi
    sample_top = top_arr[0]
    sample_bottom = bottom_arr[0]
    pred = predict_outfit_style(sample_top, sample_bottom, model_path=save_path)

    print(f"Stil Təxmini nəticəsi:")
    print(f"  Təxmin edilən Stil: {pred.style}")
    print(f"  Əminlik (Confidence): {pred.confidence:.2%}")
    print(f"  Top Stillər: {pred.top_styles}")

    assert pred.style in encoder.classes
    assert 0.0 <= pred.confidence <= 1.0
    print("✓ Task 3 Uğurlu: Style Classifier təlim və inference tam işləkdir.")


def main():
    print("Styla ML Bütün Taskların Doğrulanması Başlayır...")
    model_path = test_task_1_compatibility_training()
    test_task_2_compatibility_scoring(model_path)
    test_task_3_style_classifier()
    print("\n" + "=" * 60)
    print("BÜTÜN 3 TASK UĞURLA TAMAMLANDI VƏ TESTLƏRDƏN KEÇDİ! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
