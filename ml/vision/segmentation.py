import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import os

# Laziness load
_processor = None
_model = None

LABEL_MAP = {
    "t-shirt": 4,
    "shirt": 4,
    "jacket": 4,
    "hoodie": 4,
    "blazer": 4,
    "pants": 6,
    "jeans": 6,
    "shorts": 6,
    "skirt": 5,
    "dress": 7,
    "top": 4,
    "bottom": 6
}

def _ensure_loaded():
    global _processor, _model
    if _processor is None:
        print("[Segformer] Loading SegformerImageProcessor...")
        _processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    if _model is None:
        print("[Segformer] Loading AutoModelForSemanticSegmentation...")
        _model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        _model.eval()

def extract_garment(image_path: str, category: str, output_path: str) -> bool:
    """
    Kəsilmiş və ağ fona qoyulmuş paltarı çıxarır və output_path-a yadda saxlayır.
    """
    _ensure_loaded()
    
    cat_lower = category.lower()
    # Default olaraq top (4) götürək, əgər kateqoriya tapılmazsa
    label_id = LABEL_MAP.get(cat_lower, 4)
    
    if "pants" in cat_lower or "jeans" in cat_lower or "bottom" in cat_lower:
        label_id = 6
        
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = _processor(images=img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = _model(**inputs)
            logits = outputs.logits
            
        up = torch.nn.functional.interpolate(
            logits, size=img.size[::-1], mode="bilinear", align_corners=False
        )
        seg = up.argmax(dim=1)[0].numpy()
        
        # Maskanı yaradırıq
        mask = (seg == label_id).astype(np.uint8) * 255
        
        # Əgər maskada heç nə tapılmazsa (məsələn, adam yoxdursa və ya həmin geyim yoxdursa)
        if mask.max() == 0:
            print(f"[Segformer] Geyim (label={label_id}) tapılmadı: {image_path}")
            return False
            
        rgba = np.array(img.convert("RGBA"))
        rgba[..., 3] = mask # maskadan kənarı şəffaf et
        cut = Image.fromarray(rgba)
        
        # Ağ fon yaradırıq
        white = Image.new("RGBA", cut.size, (255, 255, 255, 255))
        white.paste(cut, (0, 0), cut) # kəsilmiş şəkli ağ fona yapışdır
        
        # Boş sahələri (ağ fonu kənarlardan) kəsirik
        bbox = Image.fromarray(mask).getbbox()
        if bbox:
            final_img = white.convert("RGB").crop(bbox)
        else:
            final_img = white.convert("RGB")
            
        # Nəticəni yadda saxlayırıq
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_img.save(output_path)
        return True
        
    except Exception as e:
        print(f"[Segformer] Xəta baş verdi: {e}")
        return False
