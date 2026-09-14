"""Re-cut existing wardrobe photos down to the garment alone, and re-embed.

A wardrobe item is a single piece of clothing, so its photo should show that
piece on white — not the model wearing it. This script runs the clothes
segmentation model over every stored photo, replaces it with the garment-only
cut-out, and recomputes the embedding and attributes from that image so
search, outfit scoring and reference matching all compare like with like.

Photos that are already garment-only product shots, and those the model would
carve into scraps, are restored from the untouched original instead — always
from `data/images/original/`, so the script is safe to re-run.

    python reprocess_wardrobe.py --email stylist@gmail.com
    python reprocess_wardrobe.py --email demo@styla.app --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

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
from ml.vision.segmentation import extract_garment  # noqa: E402

IMAGES_DIR = BASE_DIR / "data" / "images"
GARMENTS_DIR = IMAGES_DIR / "garments"
BACKUP_DIR = IMAGES_DIR / "original"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", default="demo@styla.app")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    parser.add_argument("--limit", type=int, default=0, help="process at most N items")
    args = parser.parse_args()

    db = Database()
    row = db.fetchone("SELECT id FROM users WHERE email = %s", (args.email.lower(),))
    if not row:
        print(f"No account with email {args.email}")
        return 1
    user_id = row[0]

    items = db.fetchall(
        "SELECT item_id, image_path, category FROM item_embeddings "
        "WHERE user_id = %s ORDER BY created_at",
        (user_id,),
    )
    if args.limit:
        items = items[: args.limit]
    print(f"{args.email}: {len(items)} items to reprocess")
    if args.dry_run:
        return 0

    print("Loading models...")
    embedder = FashionCLIPEmbedder()
    embedder._ensure_loaded()
    categories = CategoryClassifier(embedder)
    colors = ColorClassifier(embedder)
    patterns = PatternClassifier(embedder)
    genders = GenderClassifier(embedder)
    store = PgStore(db_url=db.db_url, ensure_schema=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    GARMENTS_DIR.mkdir(parents=True, exist_ok=True)

    cut, skipped, failed = 0, 0, 0
    t0 = time.time()
    for i, (item_id, image_path, category) in enumerate(items, start=1):
        img_file = BASE_DIR / (image_path or "").lstrip("/")
        if not image_path or not img_file.exists():
            print(f"[{i}/{len(items)}] {item_id[:8]} missing file — skipped")
            failed += 1
            continue

        # Keep the untouched photo once, so this can be re-run or undone.
        backup = BACKUP_DIR / f"{item_id}.png"
        if not backup.exists():
            shutil.copyfile(img_file, backup)

        tmp_out = img_file.with_suffix(".cut.png")
        if extract_garment(str(backup), category or "", str(tmp_out)):
            shutil.move(str(tmp_out), str(img_file))
        else:
            # Cutting out would make this photo worse (already a product shot,
            # or the model fragmented it). Restore the untouched photo — a
            # previous run may have left a damaged version in place.
            if tmp_out.exists():
                tmp_out.unlink()
            shutil.copyfile(backup, img_file)
            skipped += 1
        shutil.copyfile(img_file, GARMENTS_DIR / f"{item_id}.png")

        # Re-read attributes and the embedding from the garment-only image.
        vector = embedder.embed_images([img_file])[0]
        new_category, _ = categories.classify_vector(vector)
        color, _ = colors.classify_vector(vector)
        pattern, _ = patterns.classify_vector(vector)
        gender, _ = genders.classify_vector(vector)
        store.add(
            ids=[item_id],
            vecs=[vector],
            meta=[{
                "image_path": image_path,
                "category": new_category,
                "color": color,
                "pattern": pattern,
                "gender": gender,
                "user_id": user_id,
                "source": "wardrobe",
            }],
        )
        cut += 1
        note = "" if new_category == category else f"  (category {category} -> {new_category})"
        print(f"[{i}/{len(items)}] {item_id[:8]} {color} {new_category}{note}")

    # `skipped` counts photos deliberately left whole, which are still
    # re-embedded above, so they are not failures.

    store.close()
    db.close()
    print(f"\nProcessed {cut} items: {cut - skipped} cut out, {skipped} kept whole "
          f"(already a product shot), {failed} failed, in {time.time() - t0:.0f}s. "
          f"Originals kept in {BACKUP_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
