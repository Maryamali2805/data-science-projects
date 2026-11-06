"""Central configuration — single source of truth for all reproducibility knobs."""
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_FILE    = DATA_RAW / "creditcard.csv"
FIGURES_DIR  = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR   = PROJECT_ROOT / "models"

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Split ratios (train / val / test) ────────────────────────────────────────
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20

# ── Cross-validation ─────────────────────────────────────────────────────────
CV_FOLDS = 5

# ── Cost assumptions for threshold tuning ────────────────────────────────────
# A missed fraud costs the transaction Amount (EUR).
# A false alarm costs a fixed manual-review fee (EUR).
FALSE_ALARM_COST = 10.0   # EUR per flagged-but-legitimate transaction

# ── Kaggle dataset identifier ────────────────────────────────────────────────
KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
