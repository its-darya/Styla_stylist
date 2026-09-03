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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, classifier, color_classifier, pattern_classifier, gender_classifier, store, compat_scorer, style_scorer, personal_style
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
        items.append({
            "id": row[0],
            "imageUrl": f"http://localhost:8000{row[1]}",
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
    
    if dresses:
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
        
    compat_scores = [0.0] * len(combinations)
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

