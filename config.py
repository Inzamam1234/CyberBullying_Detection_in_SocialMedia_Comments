"""
config.py — Single source of truth for all settings.
"""

import json
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
MODEL_DIR      = BASE_DIR / "models" / "checkpoints"
MODEL_PATH     = MODEL_DIR / "toxic_model1.pt"
THRESHOLD_PATH = MODEL_DIR / "thresholds.json"

# ── Hugging Face Hub (for deployment) ─────────────────────────
HF_REPO_ID = os.getenv("HF_REPO_ID", "")

# ── Model settings ────────────────────────────────────────────
BERT_MODEL_NAME = "distilbert-base-uncased"
MAX_LEN         = 128

# ── Labels ────────────────────────────────────────────────────
LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# ── Load thresholds ────────────────────────────────────────────
def load_thresholds() -> dict:
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH) as f:
            return json.load(f)
    print("[Config] WARNING: thresholds.json not found, using 0.5 defaults")
    return {label: 0.5 for label in LABEL_COLS}

THRESHOLDS = load_thresholds()

# ── API settings ──────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT  = int(os.getenv("PORT", 8000))

# ── Training settings ─────────────────────────────────────────
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
TRAIN_SPLIT  = 0.8
RANDOM_SEED  = 42