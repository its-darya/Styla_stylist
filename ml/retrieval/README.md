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
- [Stil ballandırması](#stil-ballandırması)
- [Sürət](#sürət-perfpy)
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
| `style_prompts.py` | 8 stil × 5 şablon prompt ensembling + stil embedding keşi |
| `style_scorer.py` | Zero-shot stil balı (xam cosine + softmax probs, centering) |
| `personal_style.py` | `user_style_refs` + şəxsi stil balı (şəkil↔şəkil) |
| `perf.py` | Mərhələ-mərhələ profiling + outfit tövsiyəsi benchmark-ı |
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

## Stil ballandırması

Zero-shot stil təsnifatı: 8 stil × 5 şablon (`config.STYLES`, `config.TEMPLATES`)
prompt ensembling ilə [8, 512] stil matrisinə çevrilir.

### Prompt ensembling və keş

Hər stil üçün 5 şablon ayrıca embed olunur, nəticə ortalanır və **yenidən
L2-normalize** edilir — ortalama vektorun normu 1 deyil, normalize etməsək
skalar hasil artıq cosine olmur.

Keş faylı: `data/cache/style_embs_{MODEL_VER}_{prompt_hash}.npz`, burada
`prompt_hash = sha256(MODEL_ID + STYLES + TEMPLATES)[:12]`.

| Dəyişiklik | Nəticə |
|---|---|
| `MODEL_ID` dəyişdi | hash dəyişir → yeni fayl, yenidən hesablanır |
| `MODEL_VER` dəyişdi | fayl adının prefiksi dəyişir → yenidən hesablanır |
| `STYLES` və ya `TEMPLATES` dəyişdi | hash dəyişir → yenidən hesablanır |
| Keş faylı korlandı | oxuma xətası tutulur → yenidən hesablanır |
| Fayl adı düz, məzmun uyğun deyil | npz içindəki metadata yoxlanılır → yenidən hesablanır |

Hash həm ad, həm də məzmun səviyyəsində yoxlanılır — köhnə keşi yeni modellə
istifadə etmək **səssiz** xətadır, ona görə iki qapı qoyulub.

### 8×8 stil-stil oxşarlıq matrisi

`style_embs @ style_embs.T`, FashionCLIP (`patrickjohncyh/fashion-clip`, `prompt_hash=71ac791ff3bb`):

| | casual | formal | streetwear | sporty | bohemian | romantic | edgy | vintage |
|---|---|---|---|---|---|---|---|---|
| **casual** | 1.000 | 0.876 | 0.869 | 0.886 | 0.821 | 0.858 | 0.854 | 0.867 |
| **formal** | 0.876 | 1.000 | 0.790 | 0.796 | 0.779 | 0.858 | 0.815 | 0.861 |
| **streetwear** | 0.869 | 0.790 | 1.000 | 0.859 | 0.783 | 0.784 | 0.850 | 0.833 |
| **sporty** | 0.886 | 0.796 | 0.859 | 1.000 | 0.755 | 0.787 | 0.838 | 0.806 |
| **bohemian** | 0.821 | 0.779 | 0.783 | 0.755 | 1.000 | 0.832 | 0.781 | 0.821 |
| **romantic** | 0.858 | 0.858 | 0.784 | 0.787 | 0.832 | 1.000 | 0.799 | 0.851 |
| **edgy** | 0.854 | 0.815 | 0.850 | 0.838 | 0.781 | 0.799 | 1.000 | 0.824 |
| **vintage** | 0.867 | 0.861 | 0.833 | 0.806 | 0.821 | 0.851 | 0.824 | 1.000 |

Diaqonaldan kənar: **orta 0.826**, max **0.886** (casual ↔ sporty),
min **0.755** (bohemian ↔ sporty).

**Nəticə: heç bir cüt `STYLE_COLLISION_MAX = 0.9` həddini keçmir** — 8 stilin
hamısı saxlanılır, `STYLE_FALLBACK_CANDIDATES` (minimalist, preppy, elegant,
retro) hələlik lazım deyil.

⚠️ Mütləq qiymətlərin hamısının yüksək (~0.83) olması normaldır: CLIP-in mətn
embedding-ləri dar bir konusda yerləşir, ona görə **istənilən** iki ingilis
cümləsi arasında cosine yüksək çıxır. Əhəmiyyətli olan mütləq qiymət yox,
sıralamadır — ona görə `STYLE_CENTERING` (aşağıda) tətbiq olunur.

Yoxlama:
```bash
python -m ml.retrieval.style_prompts             # keş + matris
python -m ml.retrieval.style_prompts --markdown  # bu cədvəl
python -m ml.retrieval.style_prompts --refresh   # keşi məcburi yenilə
```

### Zero-shot stil balı (`style_scorer.py`)

```python
result = StyleScorer().score_styles(img_embs)   # img_embs: [N, 512]
result["cosine"]      # [N, 8] xam cosine
result["centered"]    # [N, 8] sütun ortası çıxılmış
result["probs"]       # [N, 8] softmax
result["logit_scale"] # 99.7928
```

**Hansı balı harada işlətməli — qarışdırmaq səhv nəticə verir:**

| Sual | İşlədilən bal | Səbəb |
|---|---|---|
| "Bu köynək daha çox casual-dır, yoxsa formal?" (BİR item daxilində sıralama) | `probs` | Softmax **sətir** üzrə normalize edir |
| "Hansı köynək daha çox formal-dır?" (İKİ item-i EYNİ stil üzrə müqayisə) | **xam `cosine`** | `probs`-un hər sətri ayrıca 1-ə toplanır — sətirlərarası müqayisə mənasızdır |

Softmax temperaturu modelin öz öyrədilmiş parametridir:
`model.logit_scale.exp()` = **99.7928** (`config.CATEGORY_LOGIT_SCALE = 100.0`
təxmini dəyər idi — praktikada fərq cüzidir, amma mənbə indi modeldir).

### Centering — öncə/sonra (50 item)

`STYLE_CENTERING=True` olanda `scores -= scores.mean(axis=0, keepdims=True)`,
yəni **N item-in hamısı üzrə** sütun ortası çıxılır.

Xam cosine-in stil sütunları üzrə ortası maqnit sinfi göstərir:

| stil | orta cosine |
|---|---|
| **vintage** | **0.2142** |
| edgy | 0.1973 |
| casual | 0.1827 |
| formal | 0.1797 |
| romantic | 0.1788 |
| sporty | 0.1682 |
| streetwear | 0.1672 |
| bohemian | 0.1556 |

⚠️ **Gözləntinin əksinə, maqnit sinif `casual` deyil — `vintage`-dır.** Polyvore
nümunə dəstəsində casual yalnız 3-cü yerdədir. Səbəb yəqin ki datasetdir
(Polyvore məhsul şəkilləri), amma nəticə budur: maqnit sinfi fərz etmək yox,
ölçmək lazımdır. Centering hansı sinfin maqnit olduğunu bilmədən işləyir,
çünki hər sütunu ayrıca sıfır-ortaya gətirir.

Qalib stilin paylanması:

| stil | öncə | sonra | fərq |
|---|---|---|---|
| casual | 3 | 5 | +2 |
| formal | 3 | 8 | +5 |
| streetwear | 1 | 6 | +5 |
| sporty | 1 | 5 | +4 |
| bohemian | 1 | 5 | +4 |
| romantic | 3 | 2 | −1 |
| edgy | 11 | 10 | −1 |
| **vintage** | **27** | **9** | **−18** |

| Metrika | Öncə | Sonra |
|---|---|---|
| Paylanma std (aşağı = balanslı) | 8.42 | **2.44** |
| Normalize entropiya (1.0 = tam bərabər) | 0.677 | **0.961** |
| Qalib stili dəyişən item | — | 24/50 |

Centering-siz item-lərin **54%-i** "vintage" etiketi alırdı — bu, faktiki
təsnifat deyil, sabit meyldir. Centering-dən sonra sütun ortaları maşın
dəqiqliyi ilə sıfırdır (`~1e-8`) və paylanma demək olar bərabərdir.

⚠️ **Centering dəstədən asılıdır.** Bal artıq mütləq deyil — verilmiş N item
daxilində nisbidir. Dəstə dəyişsə ballar da dəyişir. `N=1` üçün riyazi olaraq
mənasızdır (sətir tamamilə sıfırlanır), ona görə xəbərdarlıqla avtomatik
atlanır.

```bash
python -m ml.retrieval.style_scorer --compare            # bu müqayisə
python -m ml.retrieval.style_scorer --rank-by formal     # item-lərarası, xam cosine
python -m ml.retrieval.style_scorer --image foo.jpg
```

### Şəxsi stil balı (`personal_style.py`)

İstifadəçinin referens şəkilləri ilə şəkil↔şəkil oxşarlığı — "bu nə qədər
MƏNİM zövqümdür". `style_scorer.py`-dan fərqli sual, fərqli diapazon.

```sql
CREATE TABLE user_style_refs (
    ref_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, image_path TEXT,
    embedding VECTOR(512), model_ver TEXT, created_at TIMESTAMPTZ DEFAULT now());
CREATE INDEX user_style_refs_user_id_idx ON user_style_refs (user_id);  -- B-tree
```

`item_embeddings` **qarderobdur** (sahib olduğu əşyalar); bu isə **zövq
nümunəsidir**. Vektor indeksi (ivfflat/hnsw) qəsdən yoxdur — sorğu həmişə
`WHERE user_id = %s`, vektorlar TƏK SELECT ilə çəkilib numpy-da işlənir.

`ref_id` = `sha256(user_id + image_path)[:16]` → eyni şəkli təkrar əlavə
etmək yeni sətir yaratmır (`ON CONFLICT DO UPDATE`).

**Doğrulama** (`u_dress`: 3 don referansı, 2 don qəsdən kənarda saxlanılıb):

| # | item | bal | qeyd |
|---|---|---|---|
| 1-3 | item_0025 / 0014 / 0001 | 1.0000 | referansların özü |
| 4 | **item_0038** | **0.7732** | **kənarda saxlanılan don** |
| 5 | **item_0043** | **0.7554** | **kənarda saxlanılan don** |
| 6 | item_0034 | 0.6507 | Tops |

Görmədiyi əşyalara ümumiləşir — kənarda saxlanılan hər iki don həddi keçir.

### Niyə MAX, ortalama yox

`u_multi` = **bimodal** zövq (2 don + 2 cins referansı), 3 əşya kənarda:

| Aqreqasiya | item_0038 (don) | item_0043 (don) | item_0048 (cins) | top-6-da yad əşya |
|---|---|---|---|---|
| **max** (default) | #2 | **#4** | #1 | — |
| mean_top2 | #2 | **#4** | #1 | Sweaters |
| ~~mean (hamısı)~~ | #2 | **#9** ↓ | #1 | Sweaters, Boots, Vests |

Konkret olaraq `item_0038` (don): don-referanslara **0.773**, cins-referanslara
0.688. Ortalama bunları qarışdırıb **0.655** verir — əşya heç bir klasterə
aid olmadığı üçün yox, İKİ klasterin ortasında qaldığı üçün. Nəticədə
`item_0043` 4-cü yerdən 9-cu yerə düşür və ona əvəz olaraq heç bir referansa
oxşamayan "orta" əşyalar (Sweaters, Boots, Vests) top-6-ya qalxır.

Zövq çoxmodallıdır — eyni adam həm idman, həm klassik geyinə bilər.
`PERSONAL_AGG` ilə `max` | `mean_top2` seçilir; sadə ortalama API-də yoxdur.

### Modality gap — ölçülmüş

| Bal | min | max | orta |
|---|---|---|---|
| şəkil↔**MƏTN** (stil) | 0.0830 | 0.2885 | 0.1805 |
| şəkil↔**ŞƏKİL** (şəxsi) | 0.2364 | 1.0000 | 0.5148 |

Ortalar arasında **2.9x** fərq. Ona görə hədlər də fərqlidir:
`STYLE_TEXT_THRESHOLD = 0.25` vs `PERSONAL_SIM_THRESHOLD = 0.70`.
**Bu iki ədədi müqayisə etmək olmaz.**

Birləşdirmək lazım gələrsə — şəxsi balın töhfəsi (50% ədalətli olardı):

| Üsul | şəxsi balın çəkisi |
|---|---|
| xam toplama | 70% — stil əzilir |
| sabit diapazon (0.5–0.9 fərz edilib) | 19% — şəxsi əzilir |
| **data-dan p5–p95** | **40%** ✓ |

Diapazonu gözdən yazmaq problemi həll etmir, sadəcə istiqamətini dəyişir —
`normalize_for_fusion()` üçün miqyası müşahidə olunan paylanmadan götür.

### model_ver qoruyucusu

`get_refs()` default olaraq yalnız cari `MODEL_VER`-in referanslarını çəkir.
Model dəyişəndə köhnə vektorlar **səssiz zibil** olardı:

| Sorğu | nəticə |
|---|---|
| cari model | 3 referans |
| uyğun gəlməyən model | 0 referans (zibil qaytarılmır) |
| referansı olmayan istifadəçi | bal `[0, 0, 0]` — istisna yox, yeni istifadəçi normaldır |

```bash
python -m ml.retrieval.personal_style --add-refs u1 img1.jpg img2.jpg
python -m ml.retrieval.personal_style --score u1 --top 10
python -m ml.retrieval.personal_style --info      # cədvəl + indeks yoxlaması
```

## Sürət (`perf.py`)

Ölçülən ssenari: istifadəçi bir referens şəkil verir → qarderobundan **top-5
outfit** (bir üst + bir alt + bir ayaqqabı). Dörd mərhələ `time.perf_counter()`
ilə ayrıca ölçülür: embed / DB / scoring / kombinatorika.

```bash
python -m ml.retrieval.perf --baseline --scale 2000   # sadəlövh
python -m ml.retrieval.perf --fast     --scale 2000   # optimallaşdırılmış
python -m ml.retrieval.perf --compare  --scale 2000   # hər ikisi + fərq
python -m ml.retrieval.perf --clear                   # sintetik sətirləri sil
```

`--scale N` benchmark üçün `source='perf_bench'` işarəli sintetik sətir əlavə
edir və `finally` blokunda silir — real qarderob toxunulmaz qalır.

### Ölçmə metodologiyası

`setup_*` mərhələləri (model yüklənməsi, prompt embed-ləri) **proses-başına**
xərcdir — uzun ömürlü servisdə bir dəfə olur və hər sorğuya düşmür. Hədəf
`< 2 san` yalnız **SORĞU-başına** vaxta aiddir, ona görə ikisi ayrıca
hesablanır.

⚠️ İlk ölçmədə `embed` 7.25 san (99.8%) göründü və az qala səhv nəticə
çıxarılacaqdı: `embedder` lazy yüklənir, ona görə modelin yüklənməsi `embed`
mərhələsinin içində qalmışdı. Ayrıldıqdan sonra təmiz `embed` **0.073 san**,
model yüklənməsi isə **8.1 san** (bir dəfəlik) çıxdı.

### Öncə / sonra — 2050 sətir

| mərhələ | öncə (s) | sonra (s) | sürətlənmə |
|---|---|---|---|
| DB | 0.433 | 0.204 | 2x |
| scoring (stil) | 0.016 | 0.008 | 2x |
| slot təyini | 0.019 | 0.003 | 6x |
| **kombinatorika** | **187.891** | **0.005** | **40832x** |
| **SORĞU CƏMİ** | **188.461** | **0.314** | **599x** |

| | öncə | sonra |
|---|---|---|
| Yoxlanan kombinasiya | 45,506,370 | **1,000** |
| SQL sorğu | 2,055 | **2** |
| Hədəf (< 2 san) | ✗ 94x kənar | **✓ 0.314 san** |

Nəticə eynidir: top-5 outfit-in **sıralaması bit-bə-bit eyni**, balların
maksimum fərqi `7.15e-07` (float32 yığılma sırası fərqi — sadəlövh yol
`np.dot`-u əşya-əşya, sürətli yol tam matris hasili ilə hesablayır).

### Darboğazın tapılması

Sadəlövh baseline O(N³) böyüyür:

| qarderob | kombinasiya | kombinatorika | sorğu cəmi |
|---|---|---|---|
| 50 | 768 | 0.003 s | 0.090 s |
| 550 | 887,992 | 3.091 s | 3.297 s |
| 1050 | 5,972,120 | 22.285 s | 22.616 s |
| 2050 | 45,506,370 | 187.611 s | 188.156 s |

Qarderob 2x böyüyəndə vaxt 7.2–8.4x artır — kubik.

**Dörd optimallaşdırmanın real dəyəri (2050 sətirdə):**

| Düzəliş | Qazanc | Sorğu vaxtının %-i |
|---|---|---|
| **Kateqoriya filtri + top-K namizəd** | **187.9 s** | **99.71%** |
| N+1 sorğunun ləğvi | 0.23 s | 0.23% |
| Matris hasili (Python döngüsü yox) | 0.03 s | 0.02% |

N+1-i və bütün Python döngülərini tam yox etsək, 188.5 s → 187.9 s olardı —
hələ də hədəfdən 94x kənar. **Yalnız top-K kəsimi hədəfi tutdurur.** Digər
ikisi düzgün dəyişiklikdir (N böyüdükcə əhəmiyyəti artır), amma bu miqyasda
ikinci dərəcəlidir.

**pgvector indeksi (ivfflat/hnsw) qurulmayıb** — ölçmə bunu təsdiqləyir:
DB sorğu vaxtının 0.23%-idir. ANN indeksi recall itkisi gətirib heç nə
qazandırmazdı.

### Top-K kəsimi nəticəni niyə pozmur

Outfit balı əşya ballarının **cəmidir**, ona görə ən yaxşı outfit-lər mütləq
ən yaxşı əşyalardan qurulur — `CANDIDATES_PER_CATEGORY = 10 >= top_k = 5`
şərti ilə düzgün cavab kəsilə bilməz. (Modul B-nin uyğunluq balı əlavə
olunanda bal artıq additiv olmayacaq; onda K daha böyük seçilməli və ya
kəsim yenidən qiymətləndirilməlidir.)

### Sürətli yol miqyasdan asılı deyil

| qarderob | DB | scoring | kombinatorika | kombinasiya | SORĞU |
|---|---|---|---|---|---|
| 50 | 0.006 s | 0.001 s | 0.002 s | 480 | 0.090 s |
| 2050 | 0.204 s | 0.008 s | 0.005 s | 1,000 | 0.314 s |
| 5050 | 0.509 s | 0.007 s | 0.000 s | 1,000 | 0.611 s |
| 10050 | 1.135 s | 0.010 s | 0.000 s | 1,000 | 1.234 s |

Kombinatorika artıq sabitdir (həmişə K³ = 1000). **Darboğaz yerini dəyişdi:**
10 min sətirdə DB sorğu vaxtın **92%-idir** və xətti böyüyür — 10050 × 512
float32 ≈ 20 MB şəbəkə üzərindən. Növbəti addım (lazım olsa) qarderob
matrisini yaddaşda keşləməkdir: hər sorğuda eyni data çəkilir.

### Testlər

Repoda pytest yoxdur; `scripts/smoke_test.py` konvensiyasına uyğun müstəqil
skript (yeni asılılıq əlavə edilməyib):

```bash
python -m ml.retrieval.scripts.style_smoke_test          # 36 test
python -m ml.retrieval.scripts.style_smoke_test --no-db  # DB olmadan (31)
```

| Modul | Test | Əhatə |
|---|---|---|
| `style_prompts` | 9 | prompt sırası, hash həssaslığı, embedding müqaviləsi, keşin 4 etibarsızlaşma yolu, 8×8 matris, `collisions()` |
| `style_scorer` | 9 | shape-lər, cosine == əl hesabı, probs cəmi = 1, centering, N=1 halı, softmax stabilliyi, `rank_by_style` |
| `personal_style` | 7 | `make_ref_id`, max/mean_top2 düzgünlüyü, boş referans, `normalize_for_fusion`, `_as_array` |
| `perf` | 6 | `Profile` (istisna daxil), `timed()`, slot xəritələməsi, top-K kəsiminin itkisizliyi |
| DB (pgvector) | 5 | referans roundtrip + upsert + `model_ver` filtri, ANN indeksin YOXLUĞU, **sadəlövh == sürətli**, sintetik təmizləmə |

DB əlçatmazdırsa həmin testlər **atlanır** (uğursuzluq deyil). Testlər öz
məlumatlarını təmizləyir — `__smoke_test_user__` və `perf_bench` sətirləri
`finally` blokunda silinir.

**Mutasiya ilə yoxlanılıb.** Testlərin özləri 7 qəsdən səhvlə sınaqdan
keçirilib: centering-in çıxarılması, `mean_top2` → sadə ortalama, hash
ayırıcısının silinməsi, L2 yenidən normalizasiyanın çıxarılması, `K=1`,
softmax stabilləşdirməsinin silinməsi, slotlarda təkrar kateqoriya.

Bunlardan biri **real boşluq üzə çıxardı:** `test_style_embeddings_contract`
`load_style_embeddings()`-i keşdən oxuyurdu, ona görə hesablama yolu heç vaxt
işə düşmürdü — normalizasiya səhvi (norm 0.93–0.97) testdən keçirdi. Test
`refresh=True` ilə təzə hesablamanı da yoxlayacaq şəkildə düzəldildi.

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
