import requests
import os
import tempfile
import time

# 1. Fetch image URLs
res = requests.get('https://api.escuelajs.co/api/v1/products')
images = []
if res.status_code == 200:
    data = res.json()
    clothes = [p for p in data if 'Clothes' in p['category']['name'] or 'Clothing' in p['category']['name'] or p['category']['id'] == 1]
    for p in clothes:
        for img in p['images']:
            # filter valid urls
            if img.startswith('http') and '[' not in img and '"' not in img:
                images.append(img)

print(f"Found {len(images)} images to upload.")

# We don't want to overload the backend or take 10 hours, let's limit to 50
images = list(set(images))[:50]

success = 0
for i, img_url in enumerate(images):
    print(f"[{i+1}/{len(images)}] Downloading {img_url}...")
    try:
        img_res = requests.get(img_url, timeout=10)
        if img_res.status_code != 200:
            continue
            
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(img_res.content)
            tmp_path = tmp.name
            
        print(f"[{i+1}/{len(images)}] Uploading to API...")
        with open(tmp_path, 'rb') as f:
            files = {'file': (f"seed_{i}.jpg", f, 'image/jpeg')}
            api_res = requests.post('http://localhost:8000/api/wardrobe/upload', files=files)
            
        os.remove(tmp_path)
        
        if api_res.status_code == 200:
            print(f"[{i+1}/{len(images)}] Success! -> {api_res.json().get('category')} / {api_res.json().get('gender')}")
            success += 1
        else:
            print(f"[{i+1}/{len(images)}] API Error: {api_res.text}")
            
    except Exception as e:
        print(f"[{i+1}/{len(images)}] Failed: {e}")
        
print(f"Seeding completed. Successfully added {success} items to wardrobe.")
