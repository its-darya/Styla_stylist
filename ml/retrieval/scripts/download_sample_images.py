"""Nümunə şəkilləri HuggingFace-dən endirir (Marqo/polyvore).

Streaming rejimində ilk N sətri oxuyur — bütün datasetin (onlarla GB)
endirilməsinə ehtiyac yoxdur.

Çıxış:
    data/images/item_0001.jpg ... item_00NN.jpg
    data/images/meta.json — hər şəkil üçün orijinal metadata

meta.json strukturu:
    {"dataset": ..., "count": N,
     "items": {"item_0001": {"file": "item_0001.jpg", "source_id": "100002074_1",
                             "outfit_id": "100002074", "category": "Day Dresses",
                             "text": "tibi knit long sleeve dress"}, ...}}

`outfit_id` eyni olan əşyalar eyni outfit-dəndir — metrics.py-da gold label
kimi istifadə olunur.

İstifadə:
    python -m ml.retrieval.scripts.download_sample_images --count 50
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.retrieval import config


def _outfit_id(source_id: str) -> str:
    """`100002074_1` -> `100002074`. Ayırıcı yoxdursa id olduğu kimi qalır."""
    return source_id.rsplit("_", 1)[0] if "_" in source_id else source_id


def download(
    count: int = config.SAMPLE_IMAGE_COUNT,
    out_dir: Path = config.IMAGE_DIR,
    dataset_id: str = config.SAMPLE_DATASET_ID,
    split: str = config.SAMPLE_SPLIT,
    clean: bool = False,
) -> dict:
    from datasets import load_dataset

    out_dir = Path(out_dir)
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_id, split=split, streaming=True)
    items: dict[str, dict] = {}

    iterator = iter(dataset)
    try:
        for i in range(1, count + 1):
            row = next(iterator)
            item_id = config.SAMPLE_ID_TEMPLATE.format(i)
            filename = f"{item_id}{config.SAMPLE_IMAGE_EXT}"
            image = row["image"].convert("RGB")
            image.save(out_dir / filename, quality=95)
            source_id = str(row.get("item_ID", item_id))
            items[item_id] = {
                "file": filename,
                "source_id": source_id,
                "outfit_id": _outfit_id(source_id),
                "category": row.get("category"),
                "text": row.get("text"),
                "width": image.width,
                "height": image.height,
            }
            if i % 10 == 0:
                print(f"  {i}/{count} endirildi")
    except StopIteration:
        print(f"  dataset {len(items)} sətirdə bitdi")
    finally:
        del iterator
        del dataset
        gc.collect()

    payload = {
        "dataset": dataset_id,
        "split": split,
        "count": len(items),
        "items": items,
    }
    meta_path = out_dir / Path(config.IMAGE_META_PATH).name
    meta_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Nümunə şəkilləri endir")
    parser.add_argument("--count", type=int, default=config.SAMPLE_IMAGE_COUNT)
    parser.add_argument("--out-dir", default=config.IMAGE_DIR)
    parser.add_argument("--dataset", default=config.SAMPLE_DATASET_ID)
    parser.add_argument("--split", default=config.SAMPLE_SPLIT)
    parser.add_argument("--clean", action="store_true", help="qovluğu əvvəlcə təmizlə")
    args = parser.parse_args()

    print(f"Dataset: {args.dataset} (split={args.split}), {args.count} şəkil")
    payload = download(
        count=args.count,
        out_dir=Path(args.out_dir),
        dataset_id=args.dataset,
        split=args.split,
        clean=args.clean,
    )
    print(f"\n{payload['count']} şəkil -> {args.out_dir}")

    outfits: dict[str, int] = {}
    categories: dict[str, int] = {}
    for info in payload["items"].values():
        outfits[info["outfit_id"]] = outfits.get(info["outfit_id"], 0) + 1
        categories[info["category"]] = categories.get(info["category"], 0) + 1
    print(f"Unikal outfit : {len(outfits)}")
    print(f"Unikal kateqoriya: {len(categories)}")
    print("Top kateqoriyalar: " + ", ".join(
        f"{c}({n})" for c, n in sorted(categories.items(), key=lambda x: -x[1])[:8]
    ))
    return 0


if __name__ == "__main__":
    code = main()
    # HF streaming datasets arxa fon aiohttp thread-i saxlayır; normal
    # interpretator bağlanmasında PyGILState_Release xətası verir. İş artıq
    # bitdiyi (fayllar diskə yazıldığı) üçün buferi boşaldıb dərhal çıxırıq.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
