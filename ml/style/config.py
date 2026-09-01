"""Ensemble-level Style Classifier üçün konfiqurasiya və sabitlər.

Bütün stil təsnifatı hiperparametrləri, kateqoriyalar və defolt dəyərlər.
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

DEFAULT_MODEL_PATH = OUTPUT_DIR / "style_classifier_mlp.pt"
DEFAULT_CLASSES_PATH = OUTPUT_DIR / "style_classes.json"
DEFAULT_METRICS_PATH = OUTPUT_DIR / "style_metrics.json"

# --- Stil Taksonomiyası (Taxonomy) ------------------------------------------
STYLES = [
    "casual",
    "formal",
    "streetwear",
    "sporty",
    "business",
    "bohemian",
    "minimalist",
    "vintage",
    "chic",
    "grunge",
    "romantic",
    "preppy",
]

# Zero-shot pseudo-labeling üçün CLIP stil promptları
STYLE_PROMPT_TEMPLATE = "a photo of a {} outfit, fashion style"

# --- Model Arxitekturası -----------------------------------------------------
EMB_DIM = 512  # Hər bir FashionCLIP embedding ölçüsü
# Top + Bottom concatenation (512 + 512 = 1024) və ya full interaction features
INPUT_FEATURE_MODE = "full_pair"  # "concat" (1024) | "full_pair" (2048) | "pooled" (1024)
HIDDEN_DIMS = [256, 128]
DROPOUT = 0.25
USE_BATCH_NORM = True

# --- Təlim Hiperparametrləri -------------------------------------------------
DEVICE = os.getenv("STYLA_DEVICE", "cpu")
BATCH_SIZE = int(os.getenv("STYLA_STYLE_BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("STYLA_STYLE_EPOCHS", "25"))
LEARNING_RATE = float(os.getenv("STYLA_STYLE_LR", "1e-3"))
WEIGHT_DECAY = float(os.getenv("STYLA_STYLE_WEIGHT_DECAY", "1e-4"))
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# --- Experiment Tracking (Weights & Biases) ----------------------------------
WANDB_PROJECT = os.getenv("WANDB_STYLE_PROJECT", "styla-style-classifier")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)
WANDB_MODE = os.getenv("WANDB_MODE", "offline")
