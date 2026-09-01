"""Compatibility modulu üçün konfiqurasiya və sabitlər.

Bütün kompatibillik (uyğunluq) modeli hiperparametrləri, yollar və
defolt dəyərlər burada cəmləşdirilmişdir.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Yollar ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = MODULE_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default model çəkiləri və hesabat faylları
DEFAULT_MODEL_PATH = OUTPUT_DIR / "compatibility_mlp.pt"
DEFAULT_METRICS_PATH = OUTPUT_DIR / "compatibility_metrics.json"

# --- Model arxitekturası -----------------------------------------------------
EMB_DIM = 512  # FashionCLIP embedding ölçüsü
INPUT_REPRESENTATION = "symmetric_full"  # "symmetric_full" | "concat" | "diff_prod" | "type_aware"
# symmetric_full: [|e1-e2|, e1*e2, (e1+e2)/2, cos_sim] -> dim = 512*3 + 1 = 1537
# diff_prod: [|e1-e2|, e1*e2] -> dim = 1024
# concat: [e1, e2] -> dim = 1024
HIDDEN_DIMS = [256, 64]
DROPOUT = 0.2
USE_BATCH_NORM = True

# Type-aware model üçün
CATEGORY_EMB_DIM = 64
NUM_CATEGORIES = 20

# --- Təlim (Training) hiperparametrləri ---------------------------------------
DEVICE = os.getenv("STYLA_DEVICE", "cpu")  # "cuda" | "cpu"
BATCH_SIZE = int(os.getenv("STYLA_COMPAT_BATCH_SIZE", "64"))
EPOCHS = int(os.getenv("STYLA_COMPAT_EPOCHS", "20"))
LEARNING_RATE = float(os.getenv("STYLA_COMPAT_LR", "1e-3"))
WEIGHT_DECAY = float(os.getenv("STYLA_COMPAT_WEIGHT_DECAY", "1e-4"))
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# --- Experiment Tracking (Weights & Biases) ----------------------------------
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "styla-compatibility")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)
WANDB_MODE = os.getenv("WANDB_MODE", "offline")  # "online" | "offline" | "disabled"

# --- Qiymətləndirmə hədəfləri (README-dən) ----------------------------------
TARGET_AUC = 0.80  # README: AUC >= 0.80
TARGET_ACCURACY = 0.80
