"""Retrieval metrikləri — Recall@k, Precision@k, mAP, MRR.

Bu funksiyalar SAF-dır: store, model və ya DB-dən asılı deyil. Giriş
sıralanmış id siyahısı və relevant id çoxluğudur, ona görə D modulu
(`ml/evaluate.py`) bunları birbaşa import edib öz harness-ində işlədə bilər:

    from ml.retrieval.metrics import recall_at_k, evaluate_ranking

Terminlər:
    ranked   — modelin qaytardığı id-lər, ən yaxşıdan ən pisə (təkrarsız)
    relevant — həmin sorğu üçün doğru sayılan id-lər (gold label)

İstifadə (real qiymətləndirmə, outfit_id gold label ilə):
    python -m ml.retrieval.metrics --k 1 5 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.retrieval import config

Ranked = Sequence[str]
Relevant = Iterable[str]


def _prepare(ranked: Ranked, relevant: Relevant, k: int | None) -> tuple[list[str], set[str]]:
    relevant = set(relevant)
    ranked = list(ranked)
    if k is not None:
        if k <= 0:
            raise ValueError(f"k müsbət olmalıdır, alındı {k}")
        ranked = ranked[:k]
    return ranked, relevant


# --- tək sorğu üçün metriklər --------------------------------------------
def recall_at_k(ranked: Ranked, relevant: Relevant, k: int) -> float:
    """İlk k nəticədə tapılan relevant əşyaların bütün relevantlara nisbəti.

    Relevant çoxluq boşdursa metrik təyin olunmayıb -> 0.0.
    """
    top, rel = _prepare(ranked, relevant, k)
    if not rel:
        return 0.0
    return len(set(top) & rel) / len(rel)


def precision_at_k(ranked: Ranked, relevant: Relevant, k: int) -> float:
    """İlk k nəticənin neçə faizi relevantdır. Məxrəc k-dır (nəticə sayı deyil)."""
    top, rel = _prepare(ranked, relevant, k)
    if k <= 0:
        return 0.0
    return len(set(top) & rel) / k


def hit_rate_at_k(ranked: Ranked, relevant: Relevant, k: int) -> float:
    """İlk k-da ən azı bir relevant varsa 1.0, yoxsa 0.0."""
    top, rel = _prepare(ranked, relevant, k)
    return 1.0 if set(top) & rel else 0.0


def reciprocal_rank(ranked: Ranked, relevant: Relevant, k: int | None = None) -> float:
    """İlk relevant nəticənin mövqeyinin tərs qiyməti (1-ci -> 1.0, 2-ci -> 0.5)."""
    top, rel = _prepare(ranked, relevant, k)
    for position, item_id in enumerate(top, start=1):
        if item_id in rel:
            return 1.0 / position
    return 0.0


def average_precision(ranked: Ranked, relevant: Relevant, k: int | None = None) -> float:
    """Bir sorğu üçün Average Precision.

    Hər relevant hitdə həmin nöqtədəki precision-ların ortalaması.
    Məxrəc min(len(relevant), k) — k relevant sayından kiçik olanda
    metrik süni şəkildə cəzalandırılmasın deyə.
    """
    top, rel = _prepare(ranked, relevant, k)
    if not rel:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for position, item_id in enumerate(top, start=1):
        if item_id in rel:
            hits += 1
            precision_sum += hits / position
    denominator = min(len(rel), len(top)) if top else 0
    if denominator == 0:
        return 0.0
    return precision_sum / denominator


# --- çoxlu sorğu üçün aqreqatlar -----------------------------------------
def mean_average_precision(
    rankings: Sequence[Ranked], relevants: Sequence[Relevant], k: int | None = None
) -> float:
    """Bütün sorğular üzrə AP-nin ortalaması."""
    if len(rankings) != len(relevants):
        raise ValueError(f"{len(rankings)} sıralama, {len(relevants)} relevant — uyğun deyil")
    if not rankings:
        return 0.0
    return sum(
        average_precision(r, g, k) for r, g in zip(rankings, relevants)
    ) / len(rankings)


def mrr(
    rankings: Sequence[Ranked], relevants: Sequence[Relevant], k: int | None = None
) -> float:
    """Mean Reciprocal Rank."""
    if len(rankings) != len(relevants):
        raise ValueError(f"{len(rankings)} sıralama, {len(relevants)} relevant — uyğun deyil")
    if not rankings:
        return 0.0
    return sum(
        reciprocal_rank(r, g, k) for r, g in zip(rankings, relevants)
    ) / len(rankings)


def mean_recall_at_k(
    rankings: Sequence[Ranked], relevants: Sequence[Relevant], k: int
) -> float:
    if not rankings:
        return 0.0
    return sum(recall_at_k(r, g, k) for r, g in zip(rankings, relevants)) / len(rankings)


def mean_precision_at_k(
    rankings: Sequence[Ranked], relevants: Sequence[Relevant], k: int
) -> float:
    if not rankings:
        return 0.0
    return sum(precision_at_k(r, g, k) for r, g in zip(rankings, relevants)) / len(rankings)


def evaluate_ranking(
    rankings: Sequence[Ranked],
    relevants: Sequence[Relevant],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Bütün metrikləri bir sözlükdə qaytarır (W&B/hesabat üçün hazır format).

    Açarlar: recall@k, precision@k, hit_rate@k, mAP, MRR, num_queries
    """
    relevants = [set(r) for r in relevants]
    results: dict[str, float] = {}
    for k in ks:
        results[f"recall@{k}"] = mean_recall_at_k(rankings, relevants, k)
        results[f"precision@{k}"] = mean_precision_at_k(rankings, relevants, k)
        results[f"hit_rate@{k}"] = (
            sum(hit_rate_at_k(r, g, k) for r, g in zip(rankings, relevants)) / len(rankings)
            if rankings else 0.0
        )
    results["mAP"] = mean_average_precision(rankings, relevants)
    results["MRR"] = mrr(rankings, relevants)
    results["num_queries"] = float(len(rankings))
    return results


# --- __main__: real qiymətləndirmə ---------------------------------------
def _perturb_query_image(source: Path, destination: Path) -> Path:
    """Dataset şəklindən "istifadəçi referens şəkli" simulyasiyası.

    Ölçü dəyişmə + mərkəzdən kəsim + parlaqlıq + JPEG sıxılması. Məqsəd:
    eyni əşyanın FƏRQLİ görüntüsü ilə sorğu vermək — əks halda metrik
    triviyal olur (eyni fayl -> cosine 1.0).
    """
    from PIL import Image, ImageEnhance

    image = Image.open(source).convert("RGB")

    ratio = config.EVAL_QUERY_CROP_RATIO
    width, height = image.size
    dx, dy = int(width * (1 - ratio) / 2), int(height * (1 - ratio) / 2)
    image = image.crop((dx, dy, width - dx, height - dy))

    short_side = config.EVAL_QUERY_RESIZE
    scale = short_side / min(image.size)
    image = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.BICUBIC,
    )
    image = ImageEnhance.Brightness(image).enhance(config.EVAL_QUERY_BRIGHTNESS)
    image.save(destination, format="JPEG", quality=config.EVAL_QUERY_JPEG_QUALITY)
    return destination


def evaluate_identity_retrieval(
    ks: Sequence[int] = config.EVAL_DEFAULT_KS, backend: str | None = None
) -> dict[str, float]:
    """README-dəki "Reference matching Recall@k" — ƏSAS metrik.

    Sorğu: əşyanın dəyişdirilmiş (kəsilmiş/sıxılmış/işıqlandırılmış) şəkli.
    Relevant: qarderobdakı EYNİ əşya (tək doğru cavab).
    Bu, referens outfit-dəki əşyanı istifadəçinin qarderobunda tapmaq
    məsələsinin birbaşa ölçüsüdür.
    """
    import tempfile

    from ml.retrieval.search import Searcher

    with Searcher(backend=backend) as searcher:
        store = searcher.store
        if store.count() == 0:
            raise RuntimeError("Store boşdur -> python -m ml.retrieval.ingest")
        item_ids = list(store.ids) if hasattr(store, "ids") else []
        if not item_ids:
            raise RuntimeError("Bu backend id siyahısını vermir — numpy backend işlət")

        max_k = max(ks)
        rankings, relevants = [], []
        with tempfile.TemporaryDirectory() as tmp:
            for item_id in item_ids:
                entry = store.get(item_id)
                source = Path(entry.meta["image_path"])
                if not source.exists():
                    continue
                query_path = _perturb_query_image(source, Path(tmp) / f"{item_id}.jpg")
                results = searcher.search(image=query_path, k=max_k)
                rankings.append([r.item_id for r in results])
                relevants.append({item_id})

    metrics = evaluate_ranking(rankings, relevants, ks)
    metrics["num_items"] = float(len(item_ids))
    return metrics


def _load_outfit_labels(image_dir: Path = config.IMAGE_DIR) -> dict[str, str]:
    """item_id -> outfit_id. Gold label: eyni outfit-dən olan əşyalar relevantdır."""
    meta_path = Path(image_dir) / Path(config.IMAGE_META_PATH).name
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} yoxdur -> python -m ml.retrieval.scripts.download_sample_images"
        )
    items = json.loads(meta_path.read_text(encoding="utf-8"))["items"]
    return {item_id: info["outfit_id"] for item_id, info in items.items()}


def evaluate_wardrobe_retrieval(
    ks: Sequence[int] = (1, 5, 10), backend: str | None = None
) -> dict[str, float]:
    """Şəkil->şəkil retrieval-ı outfit_id gold label ilə qiymətləndirir.

    Hər əşya sorğudur; relevant = eyni outfit-dəki DİGƏR əşyalar.
    Sorğunun özü nəticədən çıxarılır (əks halda hər sorğu 1.0 alardı).
    """
    from ml.retrieval.search import Searcher

    labels = _load_outfit_labels()
    with Searcher(backend=backend) as searcher:
        store = searcher.store
        if store.count() == 0:
            raise RuntimeError("Store boşdur -> python -m ml.retrieval.ingest")

        item_ids = [i for i in store.ids if i in labels] if hasattr(store, "ids") \
            else list(labels)
        max_k = max(ks)
        rankings, relevants = [], []
        for item_id in item_ids:
            outfit = labels[item_id]
            relevant = {i for i, o in labels.items() if o == outfit and i != item_id}
            if not relevant:
                continue  # tək əşyalı outfit — qiymətləndirməyə töhfə vermir
            vector = store.vector_of(item_id) if hasattr(store, "vector_of") else None
            if vector is None:
                results = searcher.search(
                    image=store.get(item_id).meta["image_path"],
                    k=max_k + 1, exclude=[item_id],
                )
            else:
                results = searcher.search(vector=vector, k=max_k + 1, exclude=[item_id])
            rankings.append([r.item_id for r in results])
            relevants.append(relevant)

    metrics = evaluate_ranking(rankings, relevants, ks)
    metrics["num_items"] = float(len(item_ids))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval metrikləri")
    parser.add_argument("--k", type=int, nargs="+", default=list(config.EVAL_DEFAULT_KS))
    parser.add_argument("--backend", default=None)
    parser.add_argument("--mode", choices=["identity", "outfit"], default="identity",
                        help="identity: eyni əşyanı tap (README hədəfi) | "
                             "outfit: eyni outfit-dəki digərlərini tap")
    parser.add_argument("--json", action="store_true", help="yalnız JSON çıxışı")
    args = parser.parse_args()

    if args.mode == "identity":
        metrics = evaluate_identity_retrieval(ks=args.k, backend=args.backend)
    else:
        metrics = evaluate_wardrobe_retrieval(ks=args.k, backend=args.backend)
    if args.json:
        print(json.dumps(metrics, indent=2))
        return 0

    tesvir = {
        "identity": "Sorğu: əşyanın dəyişdirilmiş şəkli | Relevant: EYNİ əşya",
        "outfit": "Sorğu: əşyanın şəkli | Relevant: eyni outfit-dəki digərləri",
    }[args.mode]
    print("=" * 62)
    print("Styla · retrieval qiymətləndirməsi")
    print(tesvir)
    print("=" * 62)
    print(f"rejim      : {args.mode}")
    print(f"backend    : {args.backend or config.BACKEND}")
    print(f"əşya sayı  : {int(metrics['num_items'])}")
    print(f"sorğu sayı : {int(metrics['num_queries'])}")
    print()
    for k in args.k:
        print(f"  Recall@{k:<3} {metrics[f'recall@{k}']:.4f}    "
              f"Precision@{k:<3} {metrics[f'precision@{k}']:.4f}    "
              f"HitRate@{k:<3} {metrics[f'hit_rate@{k}']:.4f}")
    print()
    print(f"  mAP        {metrics['mAP']:.4f}")
    print(f"  MRR        {metrics['MRR']:.4f}")
    r5 = metrics.get("recall@5")
    if r5 is not None and args.mode == "identity":
        target = config.RECALL_AT_5_TARGET
        status = "✓ hədəfə çatıb" if r5 >= target else "✗ hədəfdən aşağı"
        print(f"\nREADME hədəfi (reference matching) Recall@5 ≥ {target}: {r5:.4f} — {status}")
    elif args.mode == "outfit":
        print("\nQeyd: outfit rejimi uyğunluq (compatibility) məsələsini ölçür —")
        print("README-dəki Recall@5 ≥ 0.7 hədəfi bu rejim üçün deyil.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
