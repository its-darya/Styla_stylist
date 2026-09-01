"""Styla Â· MÉ™rkÉ™zi QiymÉ™tlÉ™ndirmÉ™ Skripti (Rol D: Data & Evaluation).

BÃ¼tÃ¼n ML modullarÄ±nÄ±n (Retrieval, Compatibility vÉ™ Style Classifier)
metriklÉ™rini hesablayÄ±r vÉ™ README-dÉ™ki hÉ™dÉ™flÉ™rlÉ™ tutuÅŸdurur:
    - Compatibility: AUC (>=0.80), Accuracy (>=0.80), F1
    - Style Classification: Macro-F1 (>=0.80), Top-1 & Top-3 Accuracy
    - Retrieval / Reference Matching: Recall@5 (>=0.70), mAP, MRR

Ä°stifadÉ™:
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
    """Compatibility modulunu qiymÉ™tlÉ™ndirir."""
    print("\n" + "=" * 60)
    print("1. COMPATIBILITY MODULUNUN QÄ°YMÆTLÆNDÄ°RÄ°LMÆSÄ°")
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
        # NÃ¼munÉ™ test cÃ¼tlÉ™ri
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
    auc_status = "âœ“ HÆDÆFÆ Ã‡ATDI" if metrics["auc"] >= auc_target else "âœ— HÆDÆFDÆN AÅžAÄžI"

    print(f"  Test NÃ¼munÉ™ SayÄ±: {int(metrics['num_samples'])}")
    print(f"  Loss             : {metrics['loss']:.4f}")
    print(f"  Accuracy         : {metrics['accuracy']:.4f}")
    print(f"  AUC              : {metrics['auc']:.4f} (HÉ™dÉ™f: >={auc_target}) -> {auc_status}")
    print(f"  F1 Score         : {metrics['f1']:.4f}")
    print(f"  Precision        : {metrics['precision']:.4f}")
    print(f"  Recall           : {metrics['recall']:.4f}")

    return metrics


def evaluate_style_module(data_path: str | None = None) -> dict[str, Any]:
    """Style Classifier modulunu qiymÉ™tlÉ™ndirir."""
    print("\n" + "=" * 60)
    print("2. STYLE CLASSIFIER MODULUNUN QÄ°YMÆTLÆNDÄ°RÄ°LMÆSÄ°")
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

    print(f"  Test NÃ¼munÉ™ SayÄ±: {int(metrics['num_samples'])}")
    print(f"  Loss             : {metrics['loss']:.4f}")
    print(f"  Top-1 Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Top-3 Accuracy   : {metrics['top3_accuracy']:.4f}")
    print(f"  Macro-F1         : {metrics['macro_f1']:.4f} (HÉ™dÉ™f: >= 0.80)")

    return metrics


def evaluate_retrieval_module() -> dict[str, Any]:
    """Retrieval modulunu qiymÉ™tlÉ™ndirir (É™gÉ™r data varsa)."""
    print("\n" + "=" * 60)
    print("3. RETRIEVAL (REFERENCE MATCHING) QÄ°YMÆTLÆNDÄ°RÄ°LMÆSÄ°")
    print("=" * 60)

    try:
        from ml.retrieval.metrics import evaluate_identity_retrieval
        from ml.retrieval import config
        metrics = evaluate_identity_retrieval()
        r5 = metrics.get("recall@5", 0.0)
        target = config.RECALL_AT_5_TARGET
        status = "âœ“ HÆDÆFÆ Ã‡ATDI" if r5 >= target else "âœ— HÆDÆFDÆN AÅžAÄžI"
        print(f"  Recall@1 : {metrics.get('recall@1', 0.0):.4f}")
        print(f"  Recall@5 : {r5:.4f} (HÉ™dÉ™f: >={target}) -> {status}")
        print(f"  Recall@10: {metrics.get('recall@10', 0.0):.4f}")
        print(f"  mAP      : {metrics.get('mAP', 0.0):.4f}")
        print(f"  MRR      : {metrics.get('MRR', 0.0):.4f}")
        return metrics
    except Exception as exc:
        print(f"  [Qeyd] Retrieval qiymÉ™tlÉ™ndirilmÉ™si keÃ§ildi ({exc}).")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Styla BÃ¼tÃ¼n ModullarÄ±n QiymÉ™tlÉ™ndirilmÉ™si")
    parser.add_argument("--all", action="store_true", help="BÃ¼tÃ¼n modullarÄ± qiymÉ™tlÉ™ndir")
    parser.add_argument("--compatibility", action="store_true", help="YalnÄ±z Compatibility modulunu")
    parser.add_argument("--style", action="store_true", help="YalnÄ±z Style Classifier modulunu")
    parser.add_argument("--retrieval", action="store_true", help="YalnÄ±z Retrieval modulunu")
    parser.add_argument("--compat-data", type=str, default=None)
    parser.add_argument("--style-data", type=str, default=None)
    args = parser.parse_args()

    # HeÃ§ bir flag verilmÉ™dikdÉ™ defolt olaraq hamÄ±sÄ±nÄ± yoxlayÄ±rÄ±q
    run_all = args.all or (not args.compatibility and not args.style and not args.retrieval)

    results = {}
    if run_all or args.compatibility:
        results["compatibility"] = evaluate_compatibility_module(args.compat_data)
    if run_all or args.style:
        results["style"] = evaluate_style_module(args.style_data)
    if run_all or args.retrieval:
        results["retrieval"] = evaluate_retrieval_module()

    print("\n" + "=" * 60)
    print("QÄ°YMÆTLÆNDÄ°RMÆ XÃœLASÆSÄ° TAMAMLANDI")
    print("=" * 60)


if __name__ == "__main__":
    main()
# ml/evaluate.py
import random

from ml.compatibility.generate import WardrobeItem, generate_outfit, outfit_is_valid
from ml.vision.classify import ClassifiedItem, filter_by_confidence


def classification_report(
    items: list[ClassifiedItem],
    labels: tuple[str, ...],
) -> dict:
    y_true = [i.true_category for i in items]
    y_pred = [i.predicted_category for i in items]

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(items) if items else 0.0

    per_class = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(labels) if labels else 0.0
    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class}


def _to_wardrobe_items(items: list[ClassifiedItem]) -> list[WardrobeItem]:
    return [
        WardrobeItem(
            item_id=item.item_id,
            category=item.predicted_category,
            true_category=item.true_category,
        )
        for item in items
    ]


def valid_outfit_rate(
    items: list[ClassifiedItem],
    *,
    n_outfits: int = 100,
    seed: int = 0,
) -> float:
    wardrobe = _to_wardrobe_items(items)
    rng = random.Random(seed)
    valid = 0
    for _ in range(n_outfits):
        if outfit_is_valid(generate_outfit(wardrobe, rng=rng), use_true=True):
            valid += 1
    return valid / n_outfits if n_outfits else 0.0


def threshold_sweep(
    items: list[ClassifiedItem],
    labels: tuple[str, ...],
    *,
    thresholds: list[float],
    n_outfits: int = 100,
    seed: int = 0,
) -> list[dict]:
    results = []
    for threshold in thresholds:
        kept = filter_by_confidence(items, threshold)
        report = classification_report(kept, labels)
        results.append({
            "threshold": threshold,
            "kept": len(kept),
            "total": len(items),
            "accuracy": report["accuracy"],
            "macro_f1": report["macro_f1"],
            "valid_outfit_rate": valid_outfit_rate(kept, n_outfits=n_outfits, seed=seed),
        })
    return results

