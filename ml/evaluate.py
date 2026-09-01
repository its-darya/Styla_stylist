"""Styla · Mərkəzi Qiymətləndirmə Skripti (Rol D: Data & Evaluation).

Bütün ML modullarının (Retrieval, Compatibility və Style Classifier)
metriklərini hesablayır və README-dəki hədəflərlə tutuşdurur:
    - Compatibility: AUC (>=0.80), Accuracy (>=0.80), F1
    - Style Classification: Macro-F1 (>=0.80), Top-1 & Top-3 Accuracy
    - Retrieval / Reference Matching: Recall@5 (>=0.70), mAP, MRR

İstifadə:
    python -m ml.evaluate --all
    python -m ml.evaluate --compatibility
    python -m ml.evaluate --style
    python -m ml.evaluate --retrieval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))


def evaluate_compatibility_module(data_path: str | None = None) -> dict[str, Any]:
    """Compatibility modulunu qiymətləndirir."""
    print("\n" + "=" * 60)
    print("1. COMPATIBILITY MODULUNUN QİYMƏTLƏNDİRİLMƏSİ")
    print("=" * 60)

    from ml.compatibility import config, get_scorer, score_compatibility
    from ml.compatibility.dataset import (
        PairCompatibilityDataset,
        load_pairs_from_npz,
        split_dataset,
        get_dataloaders,
    )
    from ml.compatibility.train import evaluate
    import torch
    import torch.nn as nn

    scorer = get_scorer()
    model = scorer.model

    if data_path and Path(data_path).exists():
        e1, e2, labels = load_pairs_from_npz(data_path)
    else:
        # Nümunə test cütləri
        import numpy as np
        np.random.seed(42)
        n = 200
        p_e1 = np.random.randn(n // 2, 512).astype(np.float32)
        p_e1 /= np.linalg.norm(p_e1, axis=1, keepdims=True)
        p_e2 = p_e1 + 0.2 * np.random.randn(n // 2, 512).astype(np.float32)
        p_e2 /= np.linalg.norm(p_e2, axis=1, keepdims=True)
        n_e1 = np.random.randn(n // 2, 512).astype(np.float32)
        n_e1 /= np.linalg.norm(n_e1, axis=1, keepdims=True)
        n_e2 = np.random.randn(n // 2, 512).astype(np.float32)
        n_e2 /= np.linalg.norm(n_e2, axis=1, keepdims=True)
        e1 = np.vstack([p_e1, n_e1])
        e2 = np.vstack([p_e2, n_e2])
        labels = np.concatenate([np.ones(n // 2), np.zeros(n // 2)]).astype(np.float32)

    test_ds = PairCompatibilityDataset(e1, e2, labels)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    metrics = evaluate(model, test_loader, nn.BCEWithLogitsLoss(), scorer.device)

    auc_target = config.TARGET_AUC
    auc_status = "✓ HƏDƏFƏ ÇATDI" if metrics["auc"] >= auc_target else "✗ HƏDƏFDƏN AŞAĞI"

    print(f"  Test Nümunə Sayı: {int(metrics['num_samples'])}")
    print(f"  Loss             : {metrics['loss']:.4f}")
    print(f"  Accuracy         : {metrics['accuracy']:.4f}")
    print(f"  AUC              : {metrics['auc']:.4f} (Hədəf: >={auc_target}) -> {auc_status}")
    print(f"  F1 Score         : {metrics['f1']:.4f}")
    print(f"  Precision        : {metrics['precision']:.4f}")
    print(f"  Recall           : {metrics['recall']:.4f}")

    return metrics


def evaluate_style_module(data_path: str | None = None) -> dict[str, Any]:
    """Style Classifier modulunu qiymətləndirir."""
    print("\n" + "=" * 60)
    print("2. STYLE CLASSIFIER MODULUNUN QİYMƏTLƏNDİRİLMƏSİ")
    print("=" * 60)

    from ml.style import config, get_style_classifier
    from ml.style.dataset import OutfitStyleDataset, load_style_data_from_npz
    from ml.style.train import evaluate_style
    import torch
    import torch.nn as nn

    classifier = get_style_classifier()
    model = classifier.model
    encoder = classifier.encoder

    if data_path and Path(data_path).exists():
        top_embs, bottom_embs, labels, _ = load_style_data_from_npz(data_path)
    else:
        import numpy as np
        np.random.seed(42)
        n = 200
        num_classes = encoder.num_classes()
        centers = np.random.randn(num_classes, 512).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        top_l, bot_l, lbl_l = [], [], []
        for i in range(n):
            c_idx = i % num_classes
            c = centers[c_idx]
            t = c + 0.3 * np.random.randn(512).astype(np.float32)
            b = c + 0.3 * np.random.randn(512).astype(np.float32)
            top_l.append(t / np.linalg.norm(t))
            bot_l.append(b / np.linalg.norm(b))
            lbl_l.append(c_idx)
        top_embs = np.array(top_l, dtype=np.float32)
        bottom_embs = np.array(bot_l, dtype=np.float32)
        labels = np.array(lbl_l)

    test_ds = OutfitStyleDataset(top_embs, bottom_embs, labels, encoder=encoder)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    metrics = evaluate_style(model, test_loader, nn.CrossEntropyLoss(), classifier.device)

    print(f"  Test Nümunə Sayı: {int(metrics['num_samples'])}")
    print(f"  Loss             : {metrics['loss']:.4f}")
    print(f"  Top-1 Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Top-3 Accuracy   : {metrics['top3_accuracy']:.4f}")
    print(f"  Macro-F1         : {metrics['macro_f1']:.4f} (Hədəf: >= 0.80)")

    return metrics


def evaluate_retrieval_module() -> dict[str, Any]:
    """Retrieval modulunu qiymətləndirir (əgər data varsa)."""
    print("\n" + "=" * 60)
    print("3. RETRIEVAL (REFERENCE MATCHING) QİYMƏTLƏNDİRİLMƏSİ")
    print("=" * 60)

    try:
        from ml.retrieval.metrics import evaluate_identity_retrieval
        from ml.retrieval import config
        metrics = evaluate_identity_retrieval()
        r5 = metrics.get("recall@5", 0.0)
        target = config.RECALL_AT_5_TARGET
        status = "✓ HƏDƏFƏ ÇATDI" if r5 >= target else "✗ HƏDƏFDƏN AŞAĞI"
        print(f"  Recall@1 : {metrics.get('recall@1', 0.0):.4f}")
        print(f"  Recall@5 : {r5:.4f} (Hədəf: >={target}) -> {status}")
        print(f"  Recall@10: {metrics.get('recall@10', 0.0):.4f}")
        print(f"  mAP      : {metrics.get('mAP', 0.0):.4f}")
        print(f"  MRR      : {metrics.get('MRR', 0.0):.4f}")
        return metrics
    except Exception as exc:
        print(f"  [Qeyd] Retrieval qiymətləndirilməsi keçildi ({exc}).")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Styla Bütün Modulların Qiymətləndirilməsi")
    parser.add_argument("--all", action="store_true", help="Bütün modulları qiymətləndir")
    parser.add_argument("--compatibility", action="store_true", help="Yalnız Compatibility modulunu")
    parser.add_argument("--style", action="store_true", help="Yalnız Style Classifier modulunu")
    parser.add_argument("--retrieval", action="store_true", help="Yalnız Retrieval modulunu")
    parser.add_argument("--compat-data", type=str, default=None)
    parser.add_argument("--style-data", type=str, default=None)
    args = parser.parse_args()

    # Heç bir flag verilmədikdə defolt olaraq hamısını yoxlayırıq
    run_all = args.all or (not args.compatibility and not args.style and not args.retrieval)

    results = {}
    if run_all or args.compatibility:
        results["compatibility"] = evaluate_compatibility_module(args.compat_data)
    if run_all or args.style:
        results["style"] = evaluate_style_module(args.style_data)
    if run_all or args.retrieval:
        results["retrieval"] = evaluate_retrieval_module()

    print("\n" + "=" * 60)
    print("QİYMƏTLƏNDİRMƏ XÜLASƏSİ TAMAMLANDI")
    print("=" * 60)


if __name__ == "__main__":
    main()
