import requests
import os
import tempfile
from datasets import load_dataset
import io

print('Loading dataset...')
ds = load_dataset('ceyda/fashion-products-small', split='train')

success = 0
target = 50

for i, item in enumerate(ds):
    if success >= target:
        break
        
    try:
        # Some items might not have 'image' depending on the dataset structure
        # ceyda/fashion-products-small usually has 'image' as a PIL object
        img = item['image']
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.convert('RGB').save(tmp.name, format='JPEG')
            tmp_path = tmp.name
            
        print(f"[{success+1}/{target}] Uploading to API...")
        with open(tmp_path, 'rb') as f:
            files = {'file': (f"seed_{i}.jpg", f, 'image/jpeg')}
            api_res = requests.post('http://localhost:8000/api/wardrobe/upload', files=files)
            
        os.remove(tmp_path)
        
        if api_res.status_code == 200:
            print(f"[{success+1}/{target}] Success! -> {api_res.json().get('category')} / {api_res.json().get('gender')}")
            success += 1
        else:
            print(f"[{success+1}/{target}] API Error: {api_res.text}")
            
    except Exception as e:
        print(f"[{success+1}/{target}] Failed: {e}")
        
print(f"Seeding completed. Successfully added {success} items to wardrobe.")
