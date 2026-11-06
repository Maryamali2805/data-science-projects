"""Load dataset and produce stratified train / val / test splits."""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_FILE, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO


def load_raw() -> pd.DataFrame:
    if not DATA_FILE.exists():
        sys.exit(
            f"ERROR: {DATA_FILE} not found.\n"
            "Run:  python -m src.download_data\n"
            "Or manually place creditcard.csv in data/raw/."
        )
    df = pd.read_csv(DATA_FILE)
    expected = {"Time", "Amount", "Class"}
    missing = expected - set(df.columns)
    if missing:
        sys.exit(f"ERROR: Expected columns missing from CSV: {missing}")
    return df


def split(df: pd.DataFrame):
    """Return (X_train, X_val, X_test, y_train, y_val, y_test).

    Splits are stratified on Class to preserve the ~0.17% fraud rate.
    Test ratio is computed relative to the leftover after removing the val set.
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # 1st split: carve out test
    test_size = TEST_RATIO
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    # 2nd split: carve val from trainval
    val_of_trainval = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_of_trainval, random_state=RANDOM_SEED, stratify=y_trainval
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def fraud_rate(y: pd.Series) -> float:
    return y.mean()
