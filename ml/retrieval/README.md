# ml/retrieval — Representation & Retrieval (Rol C)

FashionCLIP embedding-ləri, vector search (numpy + pgvector) və reference
outfit matching. Bu modul qarderobdakı əşyaları vektor kimi təmsil edir,
şəkil/mətn sorğusu ilə axtarış verir və referens outfit-in hansı əşyalarının
qarderobda **olmadığını** ("missing") aşkarlayır.

## Məzmun
- [Sürətli başlanğıc](#sürətli-başlanğıc)
- [Modullar](#modullar)
- [Embedding müqaviləsi](#embedding-müqaviləsi)
- [Vector store backend-ləri](#vector-store-backend-ləri)
- [Matcher](#matcher)
- [Metrikləri](#metriklər)
- [Nəticələr](#ölçülmüş-nəticələr)
- [Digər rollar üçün interfeys](#digər-rollar-üçün-interfeys)
- [Konfiqurasiya](#konfiqurasiya)

## Sürətli başlanğıc

```bash
# 1. Mühit (CPU wheel MÜTLƏQDİR — layihə GPU-suz işləyir)
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Model çəkiləri (~580 MB, .cache/ içinə)
python -m ml.retrieval.scripts.download_model

# 3. Smoke test — model işləyirmi
python -m ml.retrieval.scripts.smoke_test

# 4. Nümunə şəkillər (şəkliniz yoxdursa)
python -m ml.retrieval.scripts.download_sample_images --count 50

# 5. Embed + saxla
python -m ml.retrieval.ingest                       # yalnız disk (.npy + ids.json)
docker compose up -d                                # pgvector lazımdırsa
python -m ml.retrieval.ingest --backends numpy,pg   # disk + DB

# 6. Axtarış
python -m ml.retrieval.search --text "black leather boots" -k 5
python -m ml.retrieval.search --image data/images/item_0002.jpg -k 5 --exclude-self

# 7. Matcher, metriklər, demo
python -m ml.retrieval.matcher --reference-outfit 100002074 --exclude-self
python -m ml.retrieval.metrics --mode identity
python -m ml.retrieval.demo --all
```

Hər modulun işlək `__main__` bloku var — `python -m ml.retrieval.<modul> --help`.

## Modullar

| Fayl | Məsuliyyət |
|---|---|
| `config.py` | **Bütün** sabitlər. Koddan kənarda hardcoded dəyər yoxdur. |
| `embedder.py` | FashionCLIP: `embed_images(paths)`, `embed_texts(texts)`, `embed_pil_images(images)` |
| `store/base.py` | `VectorStore` ABC + `SearchResult` + `get_store()` factory |
| `store/numpy_store.py` | `.npy` + `ids.json`, exact cosine |
| `store/pg_store.py` | pgvector `VECTOR(512)`, `<=>` operatoru, exact |
| `ingest.py` | Şəkil qovluğu → embedding → store(lar) |
| `search.py` | Şəkil **və ya** mətn sorğusu, kateqoriya filtri, latency ölçümü |
| `matcher.py` | Referens outfit → qarderob uyğunluğu, "missing" aşkarlanması |
| `metrics.py` | Recall@k, Precision@k, HitRate@k, mAP, MRR (saf funksiyalar) |
| `demo.py` | Sorğu + top-5 → PNG grid + latency hesabatı |
| `scripts/` | Model endirmə, nümunə şəkillər, smoke test, backend müqayisəsi |

## Embedding müqaviləsi

Bütün modul boyu dəyişməz:

| Xüsusiyyət | Dəyər |
|---|---|
| Model | `patrickjohncyh/fashion-clip` (CLIP ViT-B/32, 151M parametr) |
| dtype | `float32` |
| shape | `[N, 512]` (`config.EMB_DIM`) |
| Normalizasiya | L2, ‖v‖₂ ≈ 1.0 |
| Device | `cpu` (sabit), `torch.set_num_threads(config.NUM_THREADS)` |
| Batch | `config.BATCH_SIZE` = 8 |

Vektorlar normalized olduğu üçün **cosine similarity = dot product**. Bütün
`score` dəyərləri *similarity*-dir (distance deyil), aralıq `[-1, 1]`.

## Vector store backend-ləri

İkisi də **exact** search edir — ANN index qurulmur (`config.BUILD_ANN_INDEX = False`).
MVP ölçüsündə exact həm kifayət qədər sürətlidir, həm də dəqiq nəticə zəmanətləyir.

```python
from ml.retrieval.store.base import get_store
store = get_store("numpy")          # və ya "pg"
store.add(ids, vectors, metas)
results = store.search(query_vector, k=5, where={"category": "Boots"})
```

**Disk formatı** (A modulu bunu DB-yə qoşulmadan oxuyur — dəyişdirməyin):

```
ml/retrieval/outputs/embeddings.npy   float32, [N, 512], L2-normalized
ml/retrieval/outputs/ids.json         {"ids": [...], "meta": {id: {...}}, "dim": 512, ...}
```
`ids[i]` ↔ `embeddings[i]` — sıra zəmanətlidir.

```python
import json, numpy as np
V = np.load("ml/retrieval/outputs/embeddings.npy")
ids = json.load(open("ml/retrieval/outputs/ids.json"))["ids"]
```

**pgvector sxemi** `init.sql`-dədir. `pg_store` meta-sında yalnız sxemdəki
sütunlar olur (`image_path, category, color, model_ver, source`) — `text` və
`outfit_id` kimi əlavə sahələr yalnız numpy backend-də/`meta.json`-dadır.
Axtarış nəticələrinə (id, score) təsiri yoxdur.

## Matcher

Referens outfit-in hər əşyası üçün:

1. **Zero-shot kateqoriya** — CLIP mətn promptları (`"a photo of a {}, a type of clothing"`),
   `config.CATEGORIES` üzrə softmax (`logit_scale = 100`).
2. **Kateqoriya filtri** — qarderobdakı kateqoriya adları ("Ankle Booties")
   mətn oxşarlığı ilə kobud etiketlərə ("boots") xəritələnir, axtarış həmin
   adlarla məhdudlaşır. İnam `config.CATEGORY_CONFIDENCE_MIN`-dən aşağıdırsa
   filtr **tətbiq olunmur**.
3. **Ehtiyat mexanizmi** — filtrli axtarış boş qayıdırsa və ya ən yaxşı bal
   threshold-dan aşağıdırsa, filtrsiz təkrar axtarılır. Səhv zero-shot təsnifat
   (məs. palto → "boots") doğru uyğunluğu gizlətməsin deyə.
4. **Qərar** — ən yaxşı cosine `config.MATCH_THRESHOLD` (0.75) -dan aşağıdırsa
   əşya `missing` işarələnir. E modulu bunlar üçün kataloqdan alternativ təklif edir.

```python
from ml.retrieval.matcher import Matcher
with Matcher() as matcher:
    report = matcher.match_outfit(["ref_1.jpg", "ref_2.jpg"], k=5)
    print(report.coverage)                    # 0.0–1.0
    for item in report.missing:
        print(item.query, item.predicted_category, item.score)
```

## Metriklər

`metrics.py` funksiyaları **saf**dır — model, store və ya DB-dən asılı deyil.
D modulu (`ml/evaluate.py`) onları birbaşa import edir:

```python
from ml.retrieval.metrics import recall_at_k, precision_at_k, mrr, \
    mean_average_precision, evaluate_ranking

recall_at_k(ranked_ids, relevant_ids, k=5)
evaluate_ranking(rankings, relevants, ks=(1, 5, 10))
# -> {"recall@5": ..., "precision@5": ..., "hit_rate@5": ..., "mAP": ..., "MRR": ...}
```

İki qiymətləndirmə rejimi (`python -m ml.retrieval.metrics --mode ...`):

| Rejim | Sorğu | Relevant | Nəyi ölçür |
|---|---|---|---|
| `identity` *(default)* | əşyanın dəyişdirilmiş şəkli | **eyni** əşya | Reference matching — README hədəfi |
| `outfit` | əşyanın şəkli | eyni outfit-dəki digərləri | Uyğunluq (A-nın sahəsi), müqayisə üçün |

`identity` rejimində sorğu şəkli kəsilir, kiçildilir, işıqlandırılır və JPEG ilə
sıxılır (`config.EVAL_QUERY_*`) — istifadəçinin telefon şəkli ilə dataset şəkli
arasındakı fərqin təxmini.

## Ölçülmüş nəticələr

50 əşyalıq Polyvore nümunəsində, CPU-da (16 core, `NUM_THREADS = 4`):

**Reference matching (identity rejimi)**

| Metrik | @1 | @5 | @10 |
|---|---|---|---|
| Recall | 0.940 | **1.000** | 1.000 |
| MRR | | 0.970 | |

README hədəfi Recall@5 ≥ 0.7 — ödənilir. **Amma bu rəqəm optimistdir:**
sorğu şəkilləri sintetik çevrilmələrlə alınıb, real telefon şəkilləri daha
çətindir. Çevrilmə gücünə görə həssaslıq:

| kəsim / parlaqlıq / JPEG | R@1 | R@5 |
|---|---|---|
| 0.88 / 1.15 / 55 | 1.000 | 1.000 |
| 0.70 / 1.30 / 35 *(default)* | 0.940 | 1.000 |
| 0.50 / 1.50 / 20 | 0.680 | 0.900 |
| 0.35 / 1.70 / 12 | 0.320 | 0.640 |
| 0.25 / 1.90 / 8 | 0.140 | 0.380 |

Real ölçü komandanın öz telefon şəkillərindən ibarət test dəsti ilə
aparılmalıdır (README-dəki ~100 şəkil).

**Backend ekvivalentliyi** — `scripts/compare_backends.py`: 50 sorğunun
hamısında top-5 id sırası eyni, maksimum score fərqi `7.7e-08` (float
yuvarlaqlaşdırması).

**Latency** (isti, 5 təkrarın medianı, hədəf < 2 san):

| Backend | Sorğu | embed | search | cəmi |
|---|---|---|---|---|
| numpy | mətn top-5 | 16 ms | 0.11 ms | **16 ms** |
| numpy | şəkil top-5 | 75 ms | 0.14 ms | **75 ms** |
| pg | mətn top-5 | 17 ms | 1.08 ms | **19 ms** |
| pg | şəkil top-5 | 88 ms | 1.16 ms | **90 ms** |

Soyuq başlanğıc (model yüklənməsi) ~3.5 san — prosesdə **bir dəfə**, sorğu
başına deyil. Xidmət kodu `Searcher` obyektini təkrar istifadə etməlidir.

Ingest: 50 şəkil / 6.7–7.3 san (~135–146 ms/şəkil).

## Digər rollar üçün interfeys

| Rol | Nə istifadə edir |
|---|---|
| **A** (compatibility) | `outputs/embeddings.npy` + `ids.json` — DB-siz, birbaşa numpy |
| **B** (vision) | `embedder.embed_pil_images(crops)` — fayla yazmadan PIL crop-ları |
| **D** (evaluation) | `metrics.py` funksiyaları — `ml/evaluate.py`-dan import |
| **E** (app) | `Searcher`, `Matcher` — FastAPI endpoint-lərindən |

## Konfiqurasiya

Bütün sabitlər `config.py`-dadır; əsasları mühit dəyişəni ilə override olunur
(`.env.example`-a bax):

| Dəyişən | Default | Təyinat |
|---|---|---|
| `STYLA_BACKEND` | `numpy` | Aktiv store |
| `STYLA_DB_URL` | `postgresql://styla:styla@localhost:5432/styla` | pgvector |
| `STYLA_BATCH_SIZE` | `8` | Inference batch |
| `STYLA_NUM_THREADS` | `4` | `torch.set_num_threads` |
| `STYLA_MATCH_THRESHOLD` | `0.75` | Missing həddi |
| `STYLA_INGEST_BACKENDS` | `numpy` | ingest hansı store-lara yazsın |
| `STYLA_MODEL_ID` | `patrickjohncyh/fashion-clip` | Model |

`python -m ml.retrieval.config` — aktiv konfiqurasiyanı JSON kimi çap edir.

## Qeydlər

- **GPU yoxdur.** torch CPU wheel-dən qurulmalıdır, əks halda ~2 GB lazımsız
  CUDA kitabxanası enir.
- **ANN index qurmayın.** Exact search MVP üçün kifayətdir və iki backend-in
  eyni nəticə verməsini zəmanətləyir.
- **`.npy` + `ids.json` formatını dəyişməyin.** A modulu onu DB-siz oxuyur.
- **Sabitləri `config.py`-da saxlayın.** Modul kodunda hardcoded dəyər olmamalıdır.
- Retrieval metrikləri burada, `metrics.py`-dadır — `ml/evaluate.py` D-nin faylıdır.
