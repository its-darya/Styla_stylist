"""Bütün retrieval sabitləri burada. Modul kodunda hardcoded dəyər olmamalıdır."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Yollar ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
IMAGE_META_PATH = IMAGE_DIR / "meta.json"
OUTPUT_DIR = MODULE_ROOT / "outputs"
MODEL_CACHE_DIR = Path(os.getenv("HF_HOME", PROJECT_ROOT / ".cache" / "huggingface"))

# numpy backend artefaktları (A modulu bunları DB-siz oxuyur)
EMB_PATH = OUTPUT_DIR / "embeddings.npy"
IDS_PATH = OUTPUT_DIR / "ids.json"

# --- Model ----------------------------------------------------------------
MODEL_ID = os.getenv("STYLA_MODEL_ID", "patrickjohncyh/fashion-clip")
MODEL_VER = os.getenv("STYLA_MODEL_VER", "fashionclip-v1")
EMB_DIM = 512
EMB_DTYPE = "float32"

# --- Runtime (CPU-only) ---------------------------------------------------
DEVICE = "cpu"
BATCH_SIZE = int(os.getenv("STYLA_BATCH_SIZE", "8"))
NUM_THREADS = int(os.getenv("STYLA_NUM_THREADS", "4"))

# --- Vector store ---------------------------------------------------------
BACKEND = os.getenv("STYLA_BACKEND", "numpy")  # "numpy" | "pg"
DB_URL = os.getenv(
    "STYLA_DB_URL", "postgresql://styla_user:styla_password@localhost:5440/styla_wardrobe"
)
EMB_TABLE = "item_embeddings"
BUILD_ANN_INDEX = False  # exact search kifayətdir — pgvector index qurulmur

# ingest hansı backend-lərə yazsın. Qayda: embedding-lər HƏM diskə (.npy+ids.json),
# HƏM DB-yə yazılmalıdır -> istehsalatda "numpy,pg".
INGEST_BACKENDS = [
    b.strip() for b in os.getenv("STYLA_INGEST_BACKENDS", "numpy").split(",") if b.strip()
]
DEFAULT_TOP_K = 5

# --- Matching -------------------------------------------------------------
MATCH_THRESHOLD = float(os.getenv("STYLA_MATCH_THRESHOLD", "0.75"))
CATEGORY_FILTER_ENABLED = True
# Zero-shot təsnifat üçün logit miqyası — CLIP-in öz `logit_scale`-i (~100).
# Cosine balları [-1,1] aralığındadır; softmax-dan əvvəl miqyaslanmalıdır.
CATEGORY_LOGIT_SCALE = 100.0
# Zero-shot kateqoriya təxmini bu həddən zəifdirsə filtr tətbiq olunmur
# (səhv kateqoriya filtri doğru uyğunluğu tamamilə gizlədə bilər).
CATEGORY_CONFIDENCE_MIN = 0.5
# Qarderob kateqoriya adları (məs. "Ankle Booties") kobud kateqoriyalara
# (məs. "boots") mətn oxşarlığı ilə uyğunlaşdırılır; bu həddən aşağısı atılır.
CATEGORY_MAP_MIN_SIM = 0.6

# Zero-shot kateqoriya təsnifatı üçün CLIP mətn promptları
CATEGORY_PROMPT_TEMPLATE = "a photo of a {}, a type of clothing"
CATEGORIES = [
    "t-shirt",
    "shirt",
    "sweater",
    "jacket",
    "coat",
    "dress",
    "pants",
    "jeans",
    "shorts",
    "skirt",
    "bag",
    "hat",
    "scarf",
    "sunglasses",
    "watch",
    "belt",
]

# --- Stil ballandırması (style scoring) -----------------------------------
# 8 stil etiketi. Dəyişdirilərsə stil embedding keşi avtomatik etibarsızlaşır
# (keş faylının adındakı prompt_hash bu siyahıdan hesablanır).
STYLES = [
    "casual",
    "formal",
    "streetwear",
    "sporty",
    "bohemian",
    "romantic",
    "edgy",
    "vintage",
]
# Ehtiyat stil namizədləri — 8x8 oxşarlıq matrisində hansısa cüt çox yaxın
# çıxarsa (bax STYLE_COLLISION_MAX) onlardan biri bunlarla əvəz olunur.
STYLE_FALLBACK_CANDIDATES = ["minimalist", "preppy", "elegant", "retro"]

# Prompt ensembling: hər stil 5 şablonla embed olunur, nəticə ortalanır.
# Tək prompt CLIP-də səs-küylüdür; ansambl stil istiqamətini sabitləşdirir.
TEMPLATES = [
    "a photo of a {} outfit",
    "a {} style clothing item",
    "{} fashion",
    "a person wearing {} clothes",
    "a {} look",
]
# İki stil vektoru bundan yaxındırsa onlar praktikada fərqlənmir -> xəbərdarlıq.
STYLE_COLLISION_MAX = 0.9

# Stil embedding keşi: data/cache/style_embs_{MODEL_VER}_{prompt_hash}.npz
STYLE_CACHE_DIR = DATA_DIR / "cache"
STYLE_CACHE_TEMPLATE = "style_embs_{model_ver}_{prompt_hash}.npz"
STYLE_CACHE_HASH_LEN = 12

# DİQQƏT: aşağıdakı iki hədd FƏRQLİ diapazonlarda yaşayır və bir-biri ilə
# müqayisə edilə BİLMƏZ (CLIP modality gap):
#   STYLE_TEXT_THRESHOLD  — şəkil↔MƏTN cosine, tipik olaraq 0.15-0.35
#   PERSONAL_SIM_THRESHOLD — şəkil↔ŞƏKİL cosine, tipik olaraq 0.5-0.9
STYLE_TEXT_THRESHOLD = float(os.getenv("STYLA_STYLE_TEXT_THRESHOLD", "0.25"))
PERSONAL_SIM_THRESHOLD = float(os.getenv("STYLA_PERSONAL_SIM_THRESHOLD", "0.70"))

# Şəxsi stil balının aqreqasiyası: "max" | "mean_top2".
# Default MAX — zövq çoxmodallıdır (eyni adam həm idman, həm klassik geyinə
# bilər), ona görə BÜTÜN referanslar üzrə ortalama hər ikisini cəzalandırır.
PERSONAL_AGG = os.getenv("STYLA_PERSONAL_AGG", "max")
PERSONAL_AGG_CHOICES = ("max", "mean_top2")

# "casual" maqnit sinifdir — demək olar hər şey ona doğru meyl edir. Sütun
# (stil) üzrə orta çıxılanda item-lər arasındakı NİSBİ fərq görünən olur.
STYLE_CENTERING = os.getenv("STYLA_STYLE_CENTERING", "1") not in ("0", "false", "False")

# --- Outfit kombinatorikası -----------------------------------------------
# Hər slotdan yalnız bu qədər namizəd götürülür -> K^3 kombinasiya.
CANDIDATES_PER_CATEGORY = int(os.getenv("STYLA_CANDIDATES_PER_CATEGORY", "10"))
# Outfit slotları -> `CATEGORIES` içindəki kobud etiketlər.
OUTFIT_SLOTS = {
    "top": ["t-shirt", "shirt", "sweater", "jacket", "coat", "dress"],
    "bottom": ["pants", "jeans", "shorts", "skirt"],
}

# --- İstifadəçi stil referansları -----------------------------------------
STYLE_REFS_TABLE = "user_style_refs"

# --- Qiymətləndirmə -------------------------------------------------------
# Referens şəkli simulyasiyası: istifadəçinin telefonla çəkdiyi/ekrandan
# götürdüyü şəkil dataset şəklindən fərqlənir (README-dəki "domain gap").
# Bu çevrilmələr həmin fərqi təxmin etmək üçündür.
EVAL_QUERY_RESIZE = 224          # px, qısa tərəf
EVAL_QUERY_JPEG_QUALITY = 35     # sıxılma artefaktları
EVAL_QUERY_CROP_RATIO = 0.70     # mərkəzdən kəsim (kadr fərqi)
EVAL_QUERY_BRIGHTNESS = 1.30     # işıqlandırma fərqi
EVAL_DEFAULT_KS = (1, 5, 10)
RECALL_AT_5_TARGET = 0.7  # README-dəki reference matching hədəfi

# --- Demo -----------------------------------------------------------------
DEMO_GRID_COLS = 6          # 1 sorğu + top-5
DEMO_FIG_DPI = 120
DEMO_THUMB_SIZE = (256, 320)
DEMO_LATENCY_RUNS = 5       # latency medianı üçün təkrar sayı
LATENCY_TARGET_SEC = 2.0

# --- Sample data ----------------------------------------------------------
SAMPLE_DATASET_ID = os.getenv("STYLA_SAMPLE_DATASET", "Marqo/polyvore")
SAMPLE_SPLIT = os.getenv("STYLA_SAMPLE_SPLIT", "data")
SAMPLE_IMAGE_COUNT = int(os.getenv("STYLA_SAMPLE_COUNT", "50"))
SAMPLE_ID_TEMPLATE = "item_{:04d}"
SAMPLE_IMAGE_EXT = ".jpg"

# Şəkil fayl uzantıları (ingest skan edərkən)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def summary() -> dict:
    """Aktiv konfiqurasiyanın qısa xülasəsi (CLI/log üçün)."""
    return {
        "MODEL_ID": MODEL_ID,
        "EMB_DIM": EMB_DIM,
        "DEVICE": DEVICE,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_THREADS": NUM_THREADS,
        "BACKEND": BACKEND,
        "DB_URL": DB_URL,
        "MATCH_THRESHOLD": MATCH_THRESHOLD,
        "BUILD_ANN_INDEX": BUILD_ANN_INDEX,
        "IMAGE_DIR": str(IMAGE_DIR),
        "EMB_PATH": str(EMB_PATH),
        "IDS_PATH": str(IDS_PATH),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(summary(), indent=2, ensure_ascii=False))
