"""Central configuration: single source of truth for all reproducibility knobs."""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_FILE    = DATA_RAW / "spam.csv"
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

# ── Kaggle dataset identifier ────────────────────────────────────────────────
KAGGLE_DATASET = "uciml/sms-spam-collection-dataset"

# ── Class labels ─────────────────────────────────────────────────────────────
SPAM_LABEL = "spam"
HAM_LABEL  = "ham"
