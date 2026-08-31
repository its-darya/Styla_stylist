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
