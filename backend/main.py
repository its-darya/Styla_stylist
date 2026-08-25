import sys
import os
import uuid
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fastapi.staticfiles import StaticFiles

# Modulları tapması üçün layihənin kök qovluğunu sys.path-a əlavə edirik
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.vision.background import remove_background
from ml.retrieval.embedder import FashionCLIPEmbedder
from ml.retrieval.matcher import CategoryClassifier, ColorClassifier, PatternClassifier
from ml.retrieval.store.pg_store import PgStore

# Qlobal ML modellər və DB bağlantısı
embedder = None
classifier = None
color_classifier = None
pattern_classifier = None
store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, classifier, color_classifier, pattern_classifier, store
    print("Initializing ML models...")
    embedder = FashionCLIPEmbedder()
    # Yükləməni tezləşdirmək üçün ilk dəfədən yükləyirik
    embedder._ensure_loaded()
    classifier = CategoryClassifier(embedder)
    color_classifier = ColorClassifier(embedder)
    pattern_classifier = PatternClassifier(embedder)
    
    print("Connecting to Vector Store (PgStore)...")
    store = PgStore(ensure_schema=True)
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
            """SELECT item_id, image_path, category, color, pattern, created_at 
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
            "dateAdded": row[5].isoformat() if row[5] else None
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

@app.post("/api/wardrobe/upload", response_model=UploadResponse)
async def upload_wardrobe_item(file: UploadFile = File(...)):
    if not embedder or not classifier or not color_classifier or not pattern_classifier or not store:
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
        
        # 5. DB-yə (pgvector) qeyd edilməsi
        meta = {
            "image_path": f"/data/images/{item_id}.png",
            "category": category,
            "color": color,
            "pattern": pattern,
            "source": "wardrobe"
        }
        
        # VectorStore.add gözləyir: ids, vecs (shape [N, 512]), metas
        store.add(ids=[item_id], vecs=[vector], meta=[meta])
        
        return UploadResponse(
            id=item_id,
            filename=f"/data/images/{item_id}.png",
            category=category,
            color=color,
            pattern=pattern
        )
    finally:
        # Təmizlik (orijinal şəkli silirik, ancaq şəffafı DB üçün saxlayırıq)
        if tmp_path.exists():
            tmp_path.unlink()
