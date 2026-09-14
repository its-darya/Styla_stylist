"""Seed a wardrobe from the HuggingFace fashion-product-images dataset.

Uploads go through the normal API, so each item is analysed and stored for
the signed-in account. Defaults to the built-in demo account.

    python seed_hf.py                       # 100 items into demo@styla.app
    python seed_hf.py --count 40 --email you@example.com --password ...
"""
import argparse
import os
import tempfile

import requests
from datasets import load_dataset

from seed_utils import login

parser = argparse.ArgumentParser()
parser.add_argument("--api", default=os.getenv("STYLA_API", "http://localhost:8000"))
parser.add_argument("--email", default=os.getenv("STYLA_DEMO_EMAIL", "demo@styla.app"))
parser.add_argument("--password", default=os.getenv("STYLA_DEMO_PASSWORD", "demo1234"))
parser.add_argument("--count", type=int, default=100)
args = parser.parse_args()

headers = login(args.api, args.email, args.password)

print("Loading fashion dataset from HuggingFace...")
ds = load_dataset("ashraq/fashion-product-images-small", split="train", streaming=True)

# Tops, bottoms and dresses only (no innerwear / swimwear / accessories).
target_subcategories = {"Topwear", "Bottomwear", "Dress"}
items_collected = []
print("Searching for apparel items...")
for sample in ds:
    if len(items_collected) >= args.count:
        break
    if sample.get("masterCategory") == "Apparel" and sample.get("subCategory") in target_subcategories:
        items_collected.append(sample)

print(f"Collected {len(items_collected)} items. Uploading as {args.email}...")
success = 0
for i, item in enumerate(items_collected):
    print(f"[{i + 1}/{len(items_collected)}] {item['gender']} {item['baseColour']} {item['articleType']}...")
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            item["image"].convert("RGB").save(tmp.name, format="JPEG")
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            res = requests.post(
                f"{args.api}/api/wardrobe/upload",
                headers=headers,
                files={"file": (f"hf_{i}.jpg", f, "image/jpeg")},
                timeout=120,
            )
        os.remove(tmp_path)
        if res.status_code == 200:
            success += 1
        else:
            print(f"  API error: {res.text}")
    except Exception as e:
        print(f"  Failed: {e}")

print(f"Seeding completed. Successfully added {success} items to {args.email}'s wardrobe.")
