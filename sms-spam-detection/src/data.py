"""Load, clean, deduplicate and split the SMS Spam Collection dataset."""
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import DATA_FILE, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO


def load_raw() -> pd.DataFrame:
    """Load spam.csv and return a clean (label, text) DataFrame.

    Drops the three unnamed filler columns that the original file contains,
    renames v1→label and v2→text, and encodes label as 0 (ham) / 1 (spam).
    """
    if not DATA_FILE.exists():
        print(
            f"ERROR: Data file not found at {DATA_FILE}\n"
            "Run:  python -m src.download_data\n"
            "or manually place spam.csv in sms-spam-detection/data/raw/",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(DATA_FILE, encoding="latin-1")

    # Drop the three mostly-empty unnamed columns
    junk = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=junk)

    df = df.rename(columns={"v1": "label", "v2": "text"})
    df["target"] = (df["label"] == "spam").astype(int)
    df = df.reset_index(drop=True)
    return df


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate texts. Returns (deduped_df, n_removed).

    Duplicates must be removed BEFORE splitting to prevent identical messages
    appearing in both train and test sets.
    """
    n_before = len(df)
    # Keep first occurrence of each (label, text) pair
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    n_removed = n_before - len(df)
    return df, n_removed


def split(df: pd.DataFrame):
    """Stratified 60/20/20 train/val/test split on the target column.

    Call on the already-deduplicated DataFrame to avoid leakage.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X = df[["text"]]
    y = df["target"]

    test_size  = TEST_RATIO
    val_frac   = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, stratify=y_trainval,
        random_state=RANDOM_SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def spam_rate(y: pd.Series) -> float:
    return float(y.mean())
