"""Seed a wardrobe with product photos from the public Platzi fake-store API.

    python seed_db.py                       # 50 items into demo@styla.app
    python seed_db.py --count 20 --email you@example.com --password ...
"""
import argparse
import os
import tempfile

import requests

from seed_utils import login

parser = argparse.ArgumentParser()
parser.add_argument("--api", default=os.getenv("STYLA_API", "http://localhost:8000"))
parser.add_argument("--email", default=os.getenv("STYLA_DEMO_EMAIL", "demo@styla.app"))
parser.add_argument("--password", default=os.getenv("STYLA_DEMO_PASSWORD", "demo1234"))
parser.add_argument("--count", type=int, default=50)
args = parser.parse_args()

headers = login(args.api, args.email, args.password)

res = requests.get("https://api.escuelajs.co/api/v1/products", timeout=30)
images = []
if res.status_code == 200:
    for p in res.json():
        name = p.get("category", {}).get("name", "")
        if "Clothes" in name or "Clothing" in name or p.get("category", {}).get("id") == 1:
            for img in p.get("images", []):
                if img.startswith("http") and "[" not in img and '"' not in img:
                    images.append(img)

images = list(dict.fromkeys(images))[: args.count]
print(f"Found {len(images)} images to upload as {args.email}.")

success = 0
for i, img_url in enumerate(images):
    print(f"[{i + 1}/{len(images)}] {img_url}")
    try:
        img_res = requests.get(img_url, timeout=15)
        if img_res.status_code != 200:
            continue
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(img_res.content)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            api_res = requests.post(
                f"{args.api}/api/wardrobe/upload",
                headers=headers,
                files={"file": (f"seed_{i}.jpg", f, "image/jpeg")},
                timeout=120,
            )
        os.remove(tmp_path)
        if api_res.status_code == 200:
            body = api_res.json()
            print(f"   -> {body.get('color')} {body.get('fineCategory')} / {body.get('gender')}")
            success += 1
        else:
            print(f"   API error: {api_res.text}")
    except Exception as e:
        print(f"   Failed: {e}")

print(f"Seeding completed. Successfully added {success} items.")
