"""Seed a user's wardrobe directly through the app's own analysis pipeline.

Runs the same steps as `POST /api/wardrobe/upload` (background removal on a
white canvas, cutting the garment away from the model wearing it, FashionCLIP
embedding and zero-shot category / colour / pattern / gender) in-process, so
it works without the API server and without knowing the account's password.

Source photos come from the HuggingFace Kaggle fashion-product catalogue at
384x512. The widely used `ashraq/fashion-product-images-small` copy of the
same catalogue is only 60x80 and looks blurry in the UI; the 900x1200 copy is
several GB to stream. Samples are stratified across tops, bottoms, dresses and
outerwear and balanced between men's and women's items.

    python seed_wardrobe.py --email stylist@gmail.com --count 100
    python seed_wardrobe.py --email demo@styla.app --count 40 --scan 3000
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path

from PIL import Image
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

# Some ML modules log in Azerbaijani; on Windows a redirected stdout defaults to
# cp1252 and those messages raise UnicodeEncodeError, which would otherwise
# abort the item being processed.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from backend.db import Database  # noqa: E402
from ml.retrieval.embedder import FashionCLIPEmbedder  # noqa: E402
from ml.retrieval.matcher import (  # noqa: E402
    CategoryClassifier,
    ColorClassifier,
    GenderClassifier,
    PatternClassifier,
)
from ml.retrieval.store.pg_store import PgStore  # noqa: E402
from ml.vision.background import remove_background  # noqa: E402

DATASET_ID = "benitomartin/fashion-product-images-small-384x512"
# Saved wardrobe photos are capped at this long side (source is 900x1200).
MAX_SIDE = 512

IMAGES_DIR = BASE_DIR / "data" / "images"
GARMENTS_DIR = IMAGES_DIR / "garments"

# articleType -> slot used for stratified sampling
SLOT_OF_ARTICLE = {
    "Tshirts": "top", "Shirts": "top", "Tops": "top", "Sweaters": "top",
    "Sweatshirts": "top", "Tunics": "top", "Kurtas": "top",
    "Jeans": "bottom", "Trousers": "bottom", "Track Pants": "bottom", "Shorts": "bottom",
    "Skirts": "bottom", "Capris": "bottom", "Leggings": "bottom",
    "Dresses": "dress",
    "Jackets": "outerwear", "Blazers": "outerwear", "Waistcoat": "outerwear",
}
DEFAULT_QUOTA = {"top": 40, "bottom": 32, "dress": 16, "outerwear": 12}


def collect_samples(count: int, scan: int, skip: int = 0):
    from datasets import load_dataset

    print("Loading fashion dataset from HuggingFace (streaming)...")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    scale = count / sum(DEFAULT_QUOTA.values())
    quota = {k: max(1, round(v * scale)) for k, v in DEFAULT_QUOTA.items()}
    # each slot is split evenly between Men and Women
    taken: Counter = Counter()
    picked = []
    n = 0
    for n, sample in enumerate(ds):
        if n >= skip + scan or len(picked) >= count:
            break
        if n < skip:  # already used by an earlier run — don't seed duplicates
            continue
        if sample.get("masterCategory") != "Apparel":
            continue
        slot = SLOT_OF_ARTICLE.get(sample.get("articleType", ""))
        gender = sample.get("gender")
        if slot is None or gender not in ("Men", "Women"):
            continue
        key = (slot, gender)
        if taken[key] >= (quota[slot] + 1) // 2:
            continue
        taken[key] += 1
        picked.append(sample)
    print(f"Collected {len(picked)} samples after scanning {n + 1}: {dict(taken)}")
    return picked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", default="demo@styla.app")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--scan", type=int, default=4000, help="dataset rows to scan at most")
    parser.add_argument("--skip", type=int, default=0,
                        help="dataset rows to skip first (use rows already seeded by an earlier run)")
    args = parser.parse_args()

    db = Database()
    row = db.fetchone("SELECT id FROM users WHERE email = %s", (args.email.lower(),))
    if not row:
        print(f"No account with email {args.email}")
        return 1
    user_id = row[0]

    samples = collect_samples(args.count, args.scan, args.skip)
    if not samples:
        print("Nothing to seed")
        return 1

    print("Loading models...")
    embedder = FashionCLIPEmbedder()
    embedder._ensure_loaded()
    categories = CategoryClassifier(embedder)
    colors = ColorClassifier(embedder)
    patterns = PatternClassifier(embedder)
    genders = GenderClassifier(embedder)
    store = PgStore(db_url=db.db_url, ensure_schema=True)
    try:
        from ml.vision.segmentation import extract_garment
    except Exception:
        extract_garment = None

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    GARMENTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="styla_seed_"))

    added = 0
    summary: Counter = Counter()
    t0 = time.time()
    for i, sample in enumerate(samples, start=1):
        item_id = str(uuid.uuid4())
        src = tmp_dir / f"{item_id}.jpg"
        out_path = IMAGES_DIR / f"{item_id}.png"
        garment_path = GARMENTS_DIR / f"{item_id}.png"
        label = f"{sample.get('gender')} {sample.get('baseColour')} {sample.get('articleType')}"
        try:
            img = sample["image"].convert("RGB")
            if max(img.size) > MAX_SIDE:
                scale = MAX_SIDE / max(img.size)
                img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
            img.save(src, format="JPEG", quality=92)
            remove_background(src, out_path, background="white")
            vector = embedder.embed_images([out_path])[0]
            category, _ = categories.classify_vector(vector)
            # Cut the garment away from the model wearing it, then re-read the
            # attributes from the garment alone (see backend/main.py upload).
            if extract_garment is not None:
                cut_path = out_path.with_suffix(".cut.png")
                if extract_garment(str(src), category, str(cut_path)):
                    shutil.move(str(cut_path), str(out_path))
                    vector = embedder.embed_images([out_path])[0]
                    category, _ = categories.classify_vector(vector)
                elif cut_path.exists():
                    cut_path.unlink()
            shutil.copyfile(out_path, garment_path)
            color, _ = colors.classify_vector(vector)
            pattern, _ = patterns.classify_vector(vector)
            gender, _ = genders.classify_vector(vector)
            store.add(
                ids=[item_id],
                vecs=[vector],
                meta=[{
                    "image_path": f"/data/images/{item_id}.png",
                    "category": category,
                    "color": color,
                    "pattern": pattern,
                    "gender": gender,
                    "user_id": user_id,
                    "source": "seed:hf",
                }],
            )
            added += 1
            summary[category] += 1
            print(f"[{i}/{len(samples)}] {label} -> {color} {category} ({gender})")
        except Exception as exc:
            print(f"[{i}/{len(samples)}] {label} FAILED: {exc}")
            for p in (out_path, garment_path):
                if p.exists():
                    p.unlink()
        finally:
            if src.exists():
                src.unlink()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    store.close()
    total = db.fetchone("SELECT count(*) FROM item_embeddings WHERE user_id = %s", (user_id,))[0]
    db.close()
    print(f"\nAdded {added}/{len(samples)} items to {args.email} in {time.time() - t0:.0f}s "
          f"(wardrobe now {total}). By category: {dict(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
