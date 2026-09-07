import sys
import os
import uuid
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import base64
import itertools
import numpy as np
from fastapi import Form
import torch


from fastapi.staticfiles import StaticFiles

# Modulları tapması üçün layihənin kök qovluğunu sys.path-a əlavə edirik
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.vision.background import remove_background
from ml.retrieval.embedder import FashionCLIPEmbedder
from ml.retrieval.matcher import CategoryClassifier, ColorClassifier, PatternClassifier, GenderClassifier
from ml.retrieval.store.pg_store import PgStore
from ml.compatibility.scorer import get_scorer as get_compat_scorer
from ml.retrieval.style_scorer import StyleScorer
from ml.retrieval.personal_style import PersonalStyle
from ml.compatibility.rules import pattern_clash
from ml.vton.catvton_pipeline import CatVTONPipeline

# Qlobal ML modellər və DB bağlantısı
embedder = None
classifier = None
color_classifier = None
pattern_classifier = None
gender_classifier = None
store = None
compat_scorer = None
style_scorer = None
personal_style = None
kolors_pipeline = None

# In-memory job cache for try-on results  { job_id -> {status, result_url, error} }
_tryon_jobs: Dict[str, Dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, classifier, color_classifier, pattern_classifier, gender_classifier, store, compat_scorer, style_scorer, personal_style, kolors_pipeline
    print("Initializing ML models...")
    embedder = FashionCLIPEmbedder()
    # Yükləməni tezləşdirmək üçün ilk dəfədən yükləyirik
    embedder._ensure_loaded()
    classifier = CategoryClassifier(embedder)
    color_classifier = ColorClassifier(embedder)
    pattern_classifier = PatternClassifier(embedder)
    gender_classifier = GenderClassifier(embedder)
    
    print("Connecting to Vector Store (PgStore)...")
    store = PgStore(ensure_schema=True)
    compat_scorer = get_compat_scorer()
    style_scorer = StyleScorer(embedder=embedder)
    personal_style = PersonalStyle(db_url=store.db_url)
    kolors_pipeline = CatVTONPipeline()
    print("Kolors Virtual Try-On pipeline ready.")
    yield
    print("Shutting down Vector Store connection...")
    if store:
        store.close()

app = FastAPI(
    title="Styla API",
    description="Backend API for Styla - AI Personal Stylist",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Xarici müraciətlər üçün data/images qovluğunu statik kimi açırıq
data_dir = BASE_DIR / "data"
data_dir.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Styla API is running"}

@app.get("/api/wardrobe")
async def get_wardrobe_items():
    if not store:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    with store.conn.cursor() as cur:
        cur.execute(
            """SELECT item_id, image_path, category, color, pattern, gender, created_at 
               FROM item_embeddings ORDER BY created_at DESC"""
        )
        rows = cur.fetchall()
        
    items = []
    for row in rows:
        items.append({
            "id": row[0],
            "imageUrl": f"http://localhost:8000{row[1]}",
            "category": row[2],
            "color": row[3],
            "pattern": row[4] or "Solid",
            "gender": row[5] or "unisex",
            "dateAdded": row[6].isoformat() if row[6] else None
        })
    return items

@app.delete("/api/wardrobe/{item_id}")
async def delete_wardrobe_item(item_id: str):
    if not store:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        store.delete(item_id)
        # Faylı da diskdən silirik
        img_path = data_dir / "images" / f"{item_id}.png"
        if img_path.exists():
            img_path.unlink()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class UploadResponse(BaseModel):
    id: str
    filename: str
    category: str
    color: str
    pattern: str
    gender: str

@app.post("/api/wardrobe/upload", response_model=UploadResponse)
async def upload_wardrobe_item(file: UploadFile = File(...)):
    if not embedder or not classifier or not color_classifier or not pattern_classifier or not gender_classifier or not store:
        raise HTTPException(status_code=500, detail="ML Models are not initialized")
        
    item_id = str(uuid.uuid4())
    
    tmp_dir = BASE_DIR / "tmp" / "upload"
    data_img_dir = BASE_DIR / "data" / "images"
    
    tmp_dir.mkdir(parents=True, exist_ok=True)
    data_img_dir.mkdir(parents=True, exist_ok=True)
    
    tmp_path = tmp_dir / f"{item_id}_{file.filename}"
    out_path = data_img_dir / f"{item_id}.png"
    
    # 1. Şəkli müvəqqəti qovluğa yükləyirik
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 2. Arxa planın (Background) silinməsi
        remove_background(tmp_path, out_path, background="transparent")
        
        # 3. Vektorun (Embedding) çıxarılması
        vector = embedder.embed_images([out_path])[0]
        
        # 4. Zero-shot kateqoriya, rəng və naxış təyini
        category, cat_prob = classifier.classify_vector(vector)
        color, col_prob = color_classifier.classify_vector(vector)
        pattern, pat_prob = pattern_classifier.classify_vector(vector)
        gender, gen_prob = gender_classifier.classify_vector(vector)
        
        # 5. Extract and cache clean garment for Virtual Try-On using Segformer
        try:
            from ml.vision.segmentation import extract_garment
            garments_dir = data_img_dir / "garments"
            garments_dir.mkdir(parents=True, exist_ok=True)
            garment_path = garments_dir / f"{item_id}.png"
            # We run it on the original tmp_path because it has the person, out_path is already transparent but might have noise
            success = extract_garment(str(tmp_path), category, str(garment_path))
            if not success:
                # Fallback to transparent output
                shutil.copyfile(out_path, garment_path)
        except Exception as e:
            print(f"[Upload] Garment extraction failed: {e}")
            shutil.copyfile(out_path, data_img_dir / "garments" / f"{item_id}.png")
        
        # 5. DB-yə (pgvector) qeyd edilməsi
        meta = {
            "image_path": f"/data/images/{item_id}.png",
            "category": category,
            "color": color,
            "pattern": pattern,
            "gender": gender,
            "source": "wardrobe"
        }
        
        # VectorStore.add gözləyir: ids, vecs (shape [N, 512]), metas
        store.add(ids=[item_id], vecs=[vector], meta=[meta])
        
        return UploadResponse(
            id=item_id,
            filename=f"/data/images/{item_id}.png",
            category=category,
            color=color,
            pattern=pattern,
            gender=gender
        )
    finally:
        # Təmizlik (orijinal şəkli silirik, ancaq şəffafı DB üçün saxlayırıq)
        if tmp_path.exists():
            tmp_path.unlink()

class GenerateRequest(BaseModel):
    style: str
    gender: Optional[str] = "any"
    user_id: Optional[str] = "default_user"
    use_personal_style: bool = False

@app.post("/api/generate")
async def generate_outfits(req: GenerateRequest):
    if not store or not compat_scorer or not style_scorer:
        raise HTTPException(status_code=500, detail="Models not initialized")
    
    with store.conn.cursor() as cur:
        cur.execute("SELECT item_id, image_path, category, color, pattern, gender, embedding, created_at FROM item_embeddings")
        rows = cur.fetchall()
        
    items = []
    for row in rows:
        item_id = row[0]
        original_url = f"http://localhost:8000{row[1]}"
        # If segmented garment exists, use it for the thumbnail
        garment_path = BASE_DIR / "data" / "images" / "garments" / f"{item_id}.png"
        image_url = f"http://localhost:8000/data/images/garments/{item_id}.png" if garment_path.exists() else original_url
        
        items.append({
            "id": item_id,
            "imageUrl": image_url,
            "category": row[2],
            "color": row[3],
            "pattern": row[4] or "Solid",
            "gender": row[5] or "unisex",
            "embedding": np.array(row[6][1:-1].split(","), dtype=np.float32) if isinstance(row[6], str) else (row[6].to_numpy() if hasattr(row[6], 'to_numpy') else np.array(row[6], dtype=np.float32)),
            "dateAdded": row[7].isoformat() if row[7] else None
        })
        
    from ml.retrieval.config import OUTFIT_SLOTS
    if req.gender and req.gender.lower() != "any":
        req_gender = req.gender.lower()
        items = [i for i in items if i.get("gender", "unisex").lower() in (req_gender, "unisex")]

    import json
    try:
        with open(BASE_DIR / "ml" / "compatibility" / "style_rules.json") as f:
            style_rules = json.load(f)
    except Exception as e:
        print(f"Warning: could not load style rules: {e}")
        style_rules = {}

    req_style = req.style.lower()
    rule = style_rules.get(req_style, {})
    allow_cat = rule.get("allow_categories", [])
    deny_cat = rule.get("deny_categories", [])
    deny_pat = rule.get("deny_patterns", [])
    pref_col = rule.get("prefer_colors", [])
    pref_pat = rule.get("prefer_patterns", [])

    filtered_items = []
    for i in items:
        cat = i["category"].lower()
        pat = i["pattern"].lower()
        
        if allow_cat and cat not in allow_cat:
            continue
        if cat in deny_cat:
            continue
        if pat in deny_pat:
            continue
            
        filtered_items.append(i)
    items = filtered_items

    tops = [i for i in items if i["category"] in OUTFIT_SLOTS["top"] and i["category"] != "dress"]
    bottoms = [i for i in items if i["category"] in OUTFIT_SLOTS["bottom"]]
    dresses = [i for i in items if i["category"] == "dress"]
    # Shoes removed completely based on user request
    
    combinations = []
    if tops and bottoms:
        for t, b in itertools.product(tops, bottoms):
            tg = t.get("gender", "unisex").lower()
            bg = b.get("gender", "unisex").lower()
            if tg == bg or tg == "unisex" or bg == "unisex":
                combinations.append((t, b))
    
    if dresses and req.style.lower() != "sporty":
        combinations.extend([(d,) for d in dresses])

    if not combinations:
        return []
        
    all_embs = np.array([i["embedding"] for i in items])
    if len(all_embs) > 0:
        style_res = style_scorer.score_styles(all_embs, centering=False)
        styles_list = style_res["styles"]
        if req.style in styles_list:
            style_idx = styles_list.index(req.style)
            item_style_scores = {i["id"]: float(style_res["cosine"][idx][style_idx]) for idx, i in enumerate(items)}
        else:
            item_style_scores = {i["id"]: 0.0 for i in items}
    else:
        item_style_scores = {}

    item_personal_scores = {}
    if req.use_personal_style and req.user_id and personal_style:
        if personal_style.count(req.user_id) > 0:
            p_scores = personal_style.personal_score(all_embs, req.user_id)
            for idx, i in enumerate(items):
                item_personal_scores[i["id"]] = float(p_scores[idx])
                    
    top_embs = []
    bot_embs = []
    valid_comb_indices = []
    for idx, c in enumerate(combinations):
        # Determine top and bottom for compatibility scoring
        if len(c) == 2:
            # (top, bottom)
            top_embs.append(c[0]["embedding"])
            bot_embs.append(c[1]["embedding"])
            valid_comb_indices.append(idx)
        # if len(c) == 1, it's a dress so no top/bottom compat score
        
    compat_scores = [1.0 if len(c) == 1 else 0.0 for c in combinations]
    if top_embs:
        with torch.no_grad():
            c_scores = compat_scorer.score_batch(np.array(top_embs), np.array(bot_embs)).tolist()
            for i, idx in enumerate(valid_comb_indices):
                compat_scores[idx] = c_scores[i]
        
    raw_s_scores = [sum(item_style_scores.get(item["id"], 0) for item in c) / len(c) if c else 0.0 for c in combinations]
    raw_p_scores = [sum(item_personal_scores.get(item["id"], 0) for item in c) / len(c) if c else 0.0 for c in combinations]
    
    def min_max_norm(scores):
        if not scores: return scores
        min_s = min(scores)
        max_s = max(scores)
        if max_s - min_s < 1e-6:
            return [0.5 for _ in scores]
        return [(s - min_s) / (max_s - min_s) for s in scores]

    norm_c_scores = min_max_norm(compat_scores)
    norm_s_scores = min_max_norm(raw_s_scores)
    norm_p_scores = min_max_norm(raw_p_scores)

    results = []
    for idx, c in enumerate(combinations):
        c_score = norm_c_scores[idx]
        s_score = norm_s_scores[idx]
        p_score = norm_p_scores[idx]
        
        if req.use_personal_style and item_personal_scores:
            total_score = 0.5 * c_score + 0.3 * s_score + 0.2 * p_score
        else:
            total_score = 0.5 * c_score + 0.5 * s_score
            
        patterns = [item["pattern"] for item in c]
        if pattern_clash(patterns):
            total_score -= 0.15
            
        for item in c:
            if pref_col and item["color"].lower() in pref_col:
                total_score += 0.1
            if pref_pat and item["pattern"].lower() in pref_pat:
                total_score += 0.1
            
        outfit_items = [{
            "id": item["id"],
            "imageUrl": item["imageUrl"],
            "category": item["category"],
            "color": item["color"],
            "pattern": item["pattern"],
            "dateAdded": item["dateAdded"]
        } for item in c]
        results.append({
            "outfit": outfit_items,
            "score": total_score
        })
        
    results.sort(key=lambda x: x["score"], reverse=True)
    top_5 = results[:5]
    
    import datetime as dt
    now_iso = dt.datetime.utcnow().isoformat() + "Z"
    return [{"id": str(uuid.uuid4()), "style": req.style, "items": r["outfit"], "createdAt": now_iso} for r in top_5]

@app.post("/api/style/personal/upload")
async def upload_personal_style(file: UploadFile = File(...), user_id: str = Form("default_user")):
    if not embedder or not personal_style:
        raise HTTPException(status_code=500, detail="Models not initialized")
    
    tmp_dir = BASE_DIR / "tmp" / "refs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    file_uuid = str(uuid.uuid4())
    tmp_path = tmp_dir / f"{file_uuid}_{file.filename}"
    out_path = tmp_dir / f"{file_uuid}_seg.png"
    
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Segment the reference image to extract clothing/person
        remove_background(tmp_path, out_path, background="transparent")
        personal_style.add_style_refs(user_id, [out_path])
        return {"success": True}
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
        if out_path.exists():
            out_path.unlink()


# ---------------------------------------------------------------------------
# Virtual Try-On endpoints (Kolors)
# ---------------------------------------------------------------------------

class TryOnRequest(BaseModel):
    person_image_b64: Optional[str] = None          # base64-encoded JPEG/PNG of the person
    outfit_id: str
    items: List[Dict[str, Any]]    # list of { id, imageUrl, category, color, ... }


@app.post("/api/tryon")
async def start_tryon(req: TryOnRequest):
    """Start a Kolors Virtual Try-On job synchronously and return the result URL."""
    if not kolors_pipeline:
        raise HTTPException(status_code=500, detail="Try-On pipeline not initialized")

    job_id = str(uuid.uuid4())

    # 1. Decode & save the person photo or use default
    avatars_dir = BASE_DIR / "data" / "avatars"
    avatars_dir.mkdir(parents=True, exist_ok=True)
    person_path = avatars_dir / f"{job_id}_person.jpg"

    if req.person_image_b64:
        try:
            img_bytes = base64.b64decode(req.person_image_b64)
            with open(person_path, "wb") as f:
                f.write(img_bytes)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}")
    else:
        outfit_gender = "male"
        for item in req.items:
            gen = item.get("gender", "").lower()
            if gen in ["women", "female"]:
                outfit_gender = "female"
                break
        
        base_img = avatars_dir / f"base_{outfit_gender}.png"
        if not base_img.exists():
            raise HTTPException(status_code=400, detail=f"Base image not found for {outfit_gender}")
        shutil.copy(str(base_img), str(person_path))

    # 2. Run Kolors try-on pipeline
    output_dir = str(BASE_DIR / "data" / "vton")
    try:
        result_url = kolors_pipeline.try_on_outfit(
            avatar_path=str(person_path),
            outfit_items=req.items,
            output_dir=output_dir,
        )
        _tryon_jobs[job_id] = {"status": "done", "result_url": result_url}
        return {"job_id": job_id, "status": "done", "result_url": result_url}
    except Exception as exc:
        _tryon_jobs[job_id] = {"status": "error", "error": str(exc)}
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # Clean up the uploaded person photo
        if person_path.exists():
            person_path.unlink()


@app.get("/api/tryon/{job_id}")
async def get_tryon_result(job_id: str):
    """Poll for a Try-On job result (kept for frontend polling compatibility)."""
    job = _tryon_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
