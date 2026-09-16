"""Styla API — FastAPI backend.

Every user-facing endpoint requires a bearer token (see `backend/auth.py`)
and only touches rows owned by that user:

    /api/auth/*                sign up / sign in / me
    /api/wardrobe              list, upload (analyse), delete garments
    /api/generate              outfit generation (backend/outfits.py)
    /api/reference/match       reference-look matching + web product search (backend/reference.py)
    /api/looks                 saved outfits
    /api/style/personal        personal-style reference photos
    /api/tryon                 virtual try-on
    /api/pinterest/*           inspiration feed (public)

Run from the project root:  uvicorn backend.main:app --reload
or from backend/:           uvicorn main:app --reload
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import sys
import urllib.request
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

# On Windows a redirected stdout defaults to cp1252, so any library that logs
# a non-ASCII character (the try-on client prints progress ticks, some ML
# modules log in Azerbaijani) raises UnicodeEncodeError mid-request and the
# request fails with a nonsense error. Force UTF-8 and never fail on a log.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from backend import auth, pinterest  # noqa: E402
from backend.auth import CurrentUser, current_user  # noqa: E402
from backend.db import Database, ensure_demo_user, ensure_schema  # noqa: E402
from backend.outfits import OutfitGenerator, row_to_item  # noqa: E402
from backend.reference import ReferenceMatcher, coarse_category  # noqa: E402
from ml.compatibility.scorer import get_scorer as get_compat_scorer  # noqa: E402
from ml.retrieval.embedder import FashionCLIPEmbedder  # noqa: E402
from ml.retrieval.matcher import (  # noqa: E402
    CategoryClassifier,
    ColorClassifier,
    GenderClassifier,
    PatternClassifier,
)
from ml.retrieval.personal_style import PersonalStyle  # noqa: E402
from ml.retrieval.store.pg_store import PgStore  # noqa: E402
from ml.vision.background import remove_background  # noqa: E402
from ml.vton.catvton_pipeline import CatVTONPipeline  # noqa: E402

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173").rstrip("/")
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
GARMENTS_DIR = IMAGES_DIR / "garments"
TMP_DIR = BASE_DIR / "tmp"


def public_url(path: str) -> str:
    """'/data/images/x.png' -> absolute URL the browser can load."""
    if not path:
        return ""
    if path.startswith("http"):
        return path
    return f"{PUBLIC_BASE_URL}{path if path.startswith('/') else '/' + path}"


# ---------------------------------------------------------------------------
# Global services (initialised once at startup)
# ---------------------------------------------------------------------------

class Services:
    db: Database
    store: PgStore
    embedder: FashionCLIPEmbedder
    categories: CategoryClassifier
    colors: ColorClassifier
    patterns: PatternClassifier
    genders: GenderClassifier
    personal_style: PersonalStyle
    generator: OutfitGenerator
    reference: ReferenceMatcher
    vton: CatVTONPipeline
    ready: bool = False


S = Services()
_tryon_jobs: Dict[str, Dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] connecting to database...")
    S.db = Database()
    ensure_schema(S.db)
    ensure_demo_user(S.db)
    auth.configure(S.db)

    print("[startup] loading FashionCLIP...")
    S.embedder = FashionCLIPEmbedder()
    S.embedder._ensure_loaded()
    S.categories = CategoryClassifier(S.embedder)
    S.colors = ColorClassifier(S.embedder)
    S.patterns = PatternClassifier(S.embedder)
    S.genders = GenderClassifier(S.embedder)

    S.store = PgStore(db_url=S.db.db_url, ensure_schema=True)
    S.personal_style = PersonalStyle(db_url=S.db.db_url, embedder=S.embedder)
    S.generator = OutfitGenerator(S.embedder, get_compat_scorer(), S.personal_style)
    S.reference = ReferenceMatcher(
        store=S.store,
        embedder=S.embedder,
        category_classifier=S.categories,
        color_classifier=S.colors,
        pattern_classifier=S.patterns,
        gender_classifier=S.genders,
        crops_dir=DATA_DIR / "reference",
        public_url=public_url,
        item_image_url=_item_image_url,
    )
    try:
        from ml.vision.segmentation import _ensure_loaded as load_segformer

        print("[startup] loading clothes segmentation model...")
        load_segformer()
    except Exception as exc:  # reference matching falls back to whole-image mode
        print(f"[startup] segmentation model unavailable: {exc}")
    S.vton = CatVTONPipeline()
    S.ready = True
    print("[startup] ready")
    yield
    S.store.close()
    S.personal_style.close()
    S.db.close()


app = FastAPI(
    title="Styla API",
    description="Backend API for Styla - AI Personal Stylist",
    version="2.0.0",
    lifespan=lifespan,
)
# A browser refuses a credentialed request whose server answers "*", so the
# allowed sites are named instead: the local dev server, whatever FRONTEND_URL
# points at, plus anything listed in ALLOWED_ORIGINS (comma-separated). The
# regex covers Netlify's per-deploy preview subdomains.
_ALLOWED_ORIGINS = {FRONTEND_URL, "http://localhost:5173", "http://127.0.0.1:5173"}
_ALLOWED_ORIGINS.update(
    origin.strip().rstrip("/")
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(o for o in _ALLOWED_ORIGINS if o),
    allow_origin_regex=r"https://[a-z0-9-]+--[a-z0-9-]+\.netlify\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
DATA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")
app.include_router(auth.router)


def _require_ready() -> None:
    if not S.ready:
        raise HTTPException(status_code=503, detail="Models are still loading, try again in a moment")


@app.get("/health")
def health_check():
    return {"status": "ok" if S.ready else "starting", "message": "Styla API is running"}


# ---------------------------------------------------------------------------
# Wardrobe
# ---------------------------------------------------------------------------

WARDROBE_COLUMNS = "item_id, category, color, pattern, gender, embedding, created_at, image_path"


def _item_image_url(item_id: str, image_path: str | None) -> str:
    garment = GARMENTS_DIR / f"{item_id}.png"
    if garment.exists():
        return public_url(f"/data/images/garments/{item_id}.png")
    return public_url(image_path or "")


def _wardrobe_rows(user_id: str) -> list[tuple]:
    return S.db.fetchall(
        f"SELECT {WARDROBE_COLUMNS} FROM item_embeddings WHERE user_id = %s ORDER BY created_at DESC",
        (user_id,),
    )


def _row_public(row: tuple) -> dict[str, Any]:
    return {
        "id": row[0],
        "imageUrl": public_url(row[7] or ""),
        "thumbnailUrl": _item_image_url(row[0], row[7]),
        "category": coarse_category(row[1]),
        "fineCategory": row[1] or "",
        "color": row[2] or "Unknown",
        "pattern": row[3] or "Solid",
        "gender": row[4] or "unisex",
        "dateAdded": row[6].isoformat() if row[6] else None,
    }


@app.get("/api/wardrobe")
def get_wardrobe_items(user: CurrentUser = Depends(current_user)):
    _require_ready()
    return [_row_public(r) for r in _wardrobe_rows(user.id)]


@app.delete("/api/wardrobe/{item_id}")
def delete_wardrobe_item(item_id: str, user: CurrentUser = Depends(current_user)):
    _require_ready()
    owner = S.db.fetchone("SELECT user_id FROM item_embeddings WHERE item_id = %s", (item_id,))
    if not owner or owner[0] != user.id:
        raise HTTPException(status_code=404, detail="Item not found")
    S.store.delete(item_id)
    for path in (IMAGES_DIR / f"{item_id}.png", GARMENTS_DIR / f"{item_id}.png"):
        if path.exists():
            path.unlink()
    return {"success": True}


class UploadResponse(BaseModel):
    id: str
    filename: str
    imageUrl: str
    thumbnailUrl: str
    category: str
    fineCategory: str
    color: str
    pattern: str
    gender: str


@app.post("/api/wardrobe/upload", response_model=UploadResponse)
async def upload_wardrobe_item(
    file: UploadFile = File(...), user: CurrentUser = Depends(current_user)
):
    _require_ready()
    item_id = str(uuid.uuid4())
    upload_dir = TMP_DIR / "upload"
    upload_dir.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    GARMENTS_DIR.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    tmp_path = upload_dir / f"{item_id}{suffix}"
    out_path = IMAGES_DIR / f"{item_id}.png"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1. Background removal — the garment on a white canvas.
        try:
            remove_background(tmp_path, out_path, background="white")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read that image: {exc}")

        # 2. First pass of attributes, needed to pick the segmentation label.
        vector = S.embedder.embed_images([out_path])[0]
        category, _ = S.categories.classify_vector(vector)

        # 3. Cut the garment away from the model wearing it. A wardrobe item is
        #    one piece of clothing, so the stored photo shows that piece alone —
        #    which also keeps it in the same visual domain as the crops the
        #    reference matcher compares it against. Photos with no recognisable
        #    garment (a flat lay the model doesn't understand) keep the plain
        #    background-removed image.
        garment_path = GARMENTS_DIR / f"{item_id}.png"
        try:
            from ml.vision.segmentation import extract_garment

            cut_path = out_path.with_suffix(".cut.png")
            if extract_garment(str(tmp_path), category, str(cut_path)):
                shutil.move(str(cut_path), str(out_path))
                # Attributes and the embedding come from the garment itself.
                vector = S.embedder.embed_images([out_path])[0]
                category, _ = S.categories.classify_vector(vector)
            elif cut_path.exists():
                cut_path.unlink()
        except Exception as exc:
            print(f"[upload] garment extraction failed: {exc}")
        shutil.copyfile(out_path, garment_path)

        color, _ = S.colors.classify_vector(vector)
        pattern, _ = S.patterns.classify_vector(vector)
        gender, _ = S.genders.classify_vector(vector)

        # 4. Persist
        image_path = f"/data/images/{item_id}.png"
        S.store.add(
            ids=[item_id],
            vecs=[vector],
            meta=[{
                "image_path": image_path,
                "category": category,
                "color": color,
                "pattern": pattern,
                "gender": gender,
                "user_id": user.id,
                "source": "wardrobe",
            }],
        )
        return UploadResponse(
            id=item_id,
            filename=image_path,
            imageUrl=public_url(image_path),
            thumbnailUrl=_item_image_url(item_id, image_path),
            category=coarse_category(category),
            fineCategory=category,
            color=color,
            pattern=pattern,
            gender=gender,
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# Outfit generation
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    style: str
    gender: Optional[str] = "any"
    use_personal_style: bool = False
    count: int = Field(default=8, ge=1, le=20)
    # "auto" follows the style's own rule; "always"/"never" override it.
    outerwear: str = Field(default="auto", pattern="^(auto|always|never)$")
    # Build every look around this wardrobe item.
    must_include: Optional[str] = None
    # Garments to leave out of the results.
    exclude_ids: List[str] = Field(default_factory=list)
    # Skip this many top-ranked looks, so asking again returns new ones.
    offset: int = Field(default=0, ge=0, le=200)


@app.post("/api/generate")
def generate_outfits(req: GenerateRequest, user: CurrentUser = Depends(current_user)):
    _require_ready()
    rows = _wardrobe_rows(user.id)
    items = [row_to_item(r, _item_image_url(r[0], r[7])) for r in rows if r[5] is not None]
    return S.generator.generate(
        items,
        req.style,
        gender=req.gender or "any",
        user_id=user.id,
        use_personal_style=req.use_personal_style,
        count=req.count,
        outerwear=req.outerwear,
        must_include=req.must_include,
        exclude_ids=req.exclude_ids,
        offset=req.offset,
    )


# ---------------------------------------------------------------------------
# Saved looks
# ---------------------------------------------------------------------------

class SaveLookRequest(BaseModel):
    id: Optional[str] = None
    style: str
    items: List[Dict[str, Any]]
    score: Optional[float] = None


def _look_public(row: tuple) -> dict[str, Any]:
    items = row[3] if isinstance(row[3], list) else json.loads(row[3])
    return {
        "id": row[0],
        "style": row[2],
        "items": items,
        "score": row[4],
        "createdAt": row[5].isoformat() if row[5] else None,
    }


@app.get("/api/looks")
def list_looks(user: CurrentUser = Depends(current_user)):
    rows = S.db.fetchall(
        "SELECT id, user_id, style, items, score, created_at FROM saved_outfits "
        "WHERE user_id = %s ORDER BY created_at DESC",
        (user.id,),
    )
    return [_look_public(r) for r in rows]


@app.post("/api/looks")
def save_look(req: SaveLookRequest, user: CurrentUser = Depends(current_user)):
    look_id = req.id or str(uuid.uuid4())
    S.db.execute(
        "INSERT INTO saved_outfits (id, user_id, style, items, score) VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (id) DO UPDATE SET style = EXCLUDED.style, items = EXCLUDED.items, score = EXCLUDED.score",
        (look_id, user.id, req.style, json.dumps(req.items), req.score),
    )
    row = S.db.fetchone(
        "SELECT id, user_id, style, items, score, created_at FROM saved_outfits WHERE id = %s",
        (look_id,),
    )
    return _look_public(row)


@app.delete("/api/looks/{look_id}")
def delete_look(look_id: str, user: CurrentUser = Depends(current_user)):
    deleted = S.db.execute(
        "DELETE FROM saved_outfits WHERE id = %s AND user_id = %s", (look_id, user.id)
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Look not found")
    return {"success": True}


# ---------------------------------------------------------------------------
# Personal style references
# ---------------------------------------------------------------------------

@app.get("/api/style/personal")
def personal_style_status(user: CurrentUser = Depends(current_user)):
    _require_ready()
    return {"count": S.personal_style.count(user.id)}


@app.post("/api/style/personal/upload")
async def upload_personal_style(
    file: UploadFile = File(...), user: CurrentUser = Depends(current_user)
):
    _require_ready()
    refs_dir = TMP_DIR / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    ref_uuid = uuid.uuid4().hex
    suffix = Path(file.filename or "ref.jpg").suffix.lower() or ".jpg"
    tmp_path = refs_dir / f"{ref_uuid}{suffix}"
    out_path = refs_dir / f"{ref_uuid}_seg.png"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        remove_background(tmp_path, out_path, background="transparent")
        S.personal_style.add_style_refs(user.id, [out_path])
        return {"success": True, "count": S.personal_style.count(user.id)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process that photo: {exc}")
    finally:
        for p in (tmp_path, out_path):
            if p.exists():
                p.unlink()


@app.delete("/api/style/personal")
def clear_personal_style(user: CurrentUser = Depends(current_user)):
    _require_ready()
    return {"deleted": S.personal_style.delete_refs(user.id)}


# ---------------------------------------------------------------------------
# Reference matching
# ---------------------------------------------------------------------------

@app.post("/api/reference/match")
async def match_reference(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    user: CurrentUser = Depends(current_user),
):
    _require_ready()
    if file is None and not image_url:
        raise HTTPException(status_code=400, detail="file or image_url required")

    ref_dir = TMP_DIR / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = ref_dir / f"{uuid.uuid4().hex}.jpg"
    try:
        if file is not None:
            with open(tmp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        else:
            req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp, open(tmp_path, "wb") as buffer:
                shutil.copyfileobj(resp, buffer)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise HTTPException(status_code=400, detail="Could not load the reference image")

    try:
        # An empty wardrobe is fine: every piece is reported as missing, and
        # the web product search still finds where to buy it.
        return S.reference.match(tmp_path, user.id, source_url=image_url or "")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# Pinterest inspiration feed (public)
# ---------------------------------------------------------------------------

@app.get("/api/pinterest/auth")
def pinterest_auth():
    url = pinterest.get_client().auth_url()
    if not url:
        raise HTTPException(status_code=400, detail="Pinterest credentials not configured")
    return RedirectResponse(url)


@app.get("/api/pinterest/callback")
def pinterest_callback(code: Optional[str] = None, error: Optional[str] = None):
    client = pinterest.get_client()
    if code:
        client.exchange_code(code)
    return RedirectResponse(f"{FRONTEND_URL}/reference?pinterest=connected")


@app.get("/api/pinterest/feed")
def pinterest_feed(page_size: int = 25):
    return pinterest.get_client().outfit_feed(page_size=page_size)


# ---------------------------------------------------------------------------
# Virtual try-on
# ---------------------------------------------------------------------------

class TryOnRequest(BaseModel):
    person_image_b64: Optional[str] = None
    outfit_id: str
    items: List[Dict[str, Any]]


@app.post("/api/tryon")
def start_tryon(req: TryOnRequest, user: CurrentUser = Depends(current_user)):
    _require_ready()
    job_id = str(uuid.uuid4())
    avatars_dir = DATA_DIR / "avatars"
    avatars_dir.mkdir(parents=True, exist_ok=True)
    person_path = avatars_dir / f"{job_id}_person.jpg"

    if req.person_image_b64:
        try:
            person_path.write_bytes(base64.b64decode(req.person_image_b64))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}")
    else:
        outfit_gender = "male"
        for item in req.items:
            if "women" in str(item.get("gender", "")).lower() or "female" in str(item.get("gender", "")).lower():
                outfit_gender = "female"
                break
        base_img = avatars_dir / f"base_{outfit_gender}.png"
        if not base_img.exists():
            raise HTTPException(status_code=400, detail=f"Base avatar not found for {outfit_gender}")
        shutil.copy(str(base_img), str(person_path))

    try:
        result_url = S.vton.try_on_outfit(
            avatar_path=str(person_path),
            outfit_items=req.items,
            output_dir=str(DATA_DIR / "vton"),
        )
        job = {
            "status": "done",
            "result_url": result_url,
            "applied": list(getattr(S.vton, "applied", [])),
            # Garments that could not be rendered at all this time.
            "skipped": list(getattr(S.vton, "skipped", [])),
            # "ai" = a hosted try-on model produced it; "preview" = drawn
            # locally because those models were out of GPU quota or asleep.
            "mode": getattr(S.vton, "mode", "ai"),
        }
        _tryon_jobs[job_id] = job
        return {"job_id": job_id, **job}
    except Exception as exc:
        _tryon_jobs[job_id] = {"status": "error", "error": str(exc)}
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if person_path.exists():
            person_path.unlink()


@app.get("/api/tryon/{job_id}")
def get_tryon_result(job_id: str, user: CurrentUser = Depends(current_user)):
    job = _tryon_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
