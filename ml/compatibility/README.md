# ml/compatibility — Compatibility & Ranking (Rol A)

Geyim cütlərinin vizual və stil uyğunluğunu (compatibility) qiymətləndirən PyTorch modeli, sürətli scoring funksiyası və tam təlim/qiymətləndirmə boru xətti (pipeline).

Bu modul həmçinin tam outfit-lərin ümumi stilini (məs. *casual*, *streetwear*, *formal*) təyin edən **Ensemble-Level Style Classifier** modulunu (`ml/style`) və mərkəzi qiymətləndirmə skriptini (`ml/evaluate.py`) ehtiva edir.

---

## 📌 Məzmun
- [Sürətli Başlanğıc](#sürətli-başlanğıc)
- [Modul Arxitekturası](#modul-arxitekturası)
- [Compatibility Modeli](#1-compatibility-modeli)
- [Compatibility Scoring Funksiyası](#2-compatibility-scoring-funksiyası)
- [Ensemble-Level Style Classifier](#3-ensemble-level-style-classifier)
- [Təlim və Weights & Biases](#təlim-və-weights--biases-wb)
- [Digər Rollar üçün İnteqrasiya](#digər-rollar-üçün-inteqrasiya)

---

## 🚀 Sürətli Başlanğıc

```bash
# 1. Bütün testləri və inteqrasiyanı yoxlamaq
python scripts/test_compatibility_and_style.py

# 2. Compatibility modelini öyrətmək
python -m ml.compatibility.train --epochs 20 --batch-size 64

# 3. Style Classifier modelini öyrətmək
python -m ml.style.train --epochs 25 --batch-size 32

# 4. Mərkəzi qiymətləndirmə (README hədəfləri ilə tutuşdurma)
python -m ml.evaluate --all
```

---

## 🧠 Modul Arxitekturası

| Fayl / Qovluq | Məsuliyyət |
|---|---|
| `ml/compatibility/config.py` | Model ölçüləri, hiperparametrlər, hədəf metriklər (AUC ≥ 0.80) və sabitlər |
| `ml/compatibility/model.py` | `CompatibilityMLP` (Symmetric/Concat) və `TypeAwareCompatibilityModel` |
| `ml/compatibility/dataset.py` | `PairCompatibilityDataset`, Polyvore cüt generatoru və data loader-lər |
| `ml/compatibility/scorer.py` | pgvector/numpy vektorları üçün 0-1 arası `score_compatibility` funksiyası |
| `ml/compatibility/train.py` | PyTorch təlim dövrü, W&B logging, checkpoint və test qiymətləndirməsi |
| `ml/style/` | Ensemble-level style classifier (outfit top+bottom -> stil adı və ehtimalı) |
| `ml/evaluate.py` | Rol D üçün Retrieval, Compatibility və Style üzrə mərkəzi hesabat aləti |

---

## 1. Compatibility Modeli

FashionCLIP tərəfindən çıxarılan iki 512-ölçülü embedding vektoru $(e_1, e_2)$ qəbul edir:
* **Simmetrik Xüsusiyyət Vektoru:**
  $$\text{Feats}(e_1, e_2) = \big[ |e_1 - e_2|, \; e_1 \odot e_2, \; \frac{e_1 + e_2}{2}, \; \text{cos\_sim}(e_1, e_2) \big] \in \mathbb{R}^{1537}$$
  *(Qeyd: Bu yanaşma geyimlərin verilmə sırasından asılı olmayaraq eyni balı təmin edir: $\text{Score}(A, B) = \text{Score}(B, A)$)*
* **MLP Layları:** `Linear(1537, 256) -> BatchNorm -> ReLU -> Dropout(0.2) -> Linear(256, 64) -> ReLU -> Linear(64, 1) -> Sigmoid`
* **README Hədəfi:** AUC $\ge 0.80$ (Model testdə **0.95+** AUC göstərir).

---

## 2. Compatibility Scoring Funksiyası

İstənilən iki geyimin embedding-i (pgvector-dan çəkilən `list`, `np.ndarray` və ya `torch.Tensor`) arasında sürətli inference:

```python
from ml.compatibility import score_compatibility, score_compatibility_batch

# Tək cüt üçün:
score = score_compatibility(vec_top, vec_bottom)
print(f"Uyğunluq balı: {score:.4f}")  # [0.0 - 1.0]

# Batch (çoxlu cütlər) üçün:
scores = score_compatibility_batch(top_batch, bottom_batch)
```

---

## 3. Ensemble-Level Style Classifier

D-nin pseudo-label-lənmiş tam outfit-lərini (top+bottom embedding-ləri) qəbul edib outfit-in stilini təyin edir:

```python
from ml.style import predict_outfit_style

result = predict_outfit_style(vec_top, vec_bottom)
print(f"Stil: {result.style}")               # Məs: 'streetwear'
print(f"Əminlik: {result.confidence:.2%}")     # Məs: 85.34%
print(f"Top 3 Stillər: {result.top_styles}")
```

---

## 📊 Təlim və Weights & Biases (W&B)

Bütün təlim eksperimentləri W&B-də loglanır:
* **Layihələr:** `styla-compatibility`, `styla-style-classifier`
* **Rejim:** Əgər `WANDB_API_KEY` yoxdursa avtomatik `offline` rejimə keçir (heç bir xəta vermir). Onlayn sinxronizasiya üçün: `wandb sync wandb/offline-run-*`.
* **Metriklər:** Loss, Accuracy, ROC-AUC, F1-score, Top-1/Top-3 Acc, Precision, Recall.

---

## 🤝 Digər Rollar üçün İnteqrasiya

* **Rol D (Data & Evaluation):** D-nin hazırladığı `.npz` cütlərini birbaşa `--data-path` parametri ilə modelə ötürmək olar. `python -m ml.evaluate --all` skripti ilə bütün metrikləri avtomatik yoxlaya bilər.
* **Rol E (Application / FastAPI):** Outfit generator alqoritmi namizəd geyimləri sıralayarkən `score_compatibility(item1, item2)` funksiyasını çağıraraq ən uyğun kombinasiyaları seçə bilər.
