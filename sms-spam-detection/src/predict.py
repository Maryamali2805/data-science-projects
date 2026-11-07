"""Predict spam/ham for a single message from the command line.

Usage:
    python -m src.predict "Win a FREE iPhone now! Call 08001234567"

Loads the trained NB model from models/naive_bayes.joblib and prints the
label and spam probability. The classification threshold stored alongside
the model (from validation-set tuning) is applied.
"""
import sys
import json
import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR


def predict_message(text: str, threshold: float | None = None) -> dict:
    """Return {'label': 'spam'|'ham', 'probability': float, 'threshold': float}."""
    model_path = MODELS_DIR / "naive_bayes.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            "Run:  python -m src.run_all  to train the model."
        )

    pipe = joblib.load(model_path)

    # Load the chosen threshold saved by run_all
    threshold_path = MODELS_DIR / "threshold.json"
    if threshold is None:
        if threshold_path.exists():
            with open(threshold_path) as f:
                threshold = json.load(f)["chosen_threshold"]
        else:
            threshold = 0.5

    # Wrap single text in a DataFrame (same shape as training data)
    X = pd.DataFrame({"text": [text]})
    prob = pipe.predict_proba(X)[:, 1][0]
    label = "spam" if prob >= threshold else "ham"
    return {"label": label, "probability": float(prob), "threshold": float(threshold)}


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m src.predict "your message here"', file=sys.stderr)
        sys.exit(1)

    message = " ".join(sys.argv[1:])
    result  = predict_message(message)
    label   = result["label"].upper()
    prob    = result["probability"]
    thresh  = result["threshold"]

    print(f"\nMessage : {message}")
    print(f"Label   : {label}")
    print(f"Spam probability : {prob:.4f}  (threshold = {thresh:.4f})")
    print()


if __name__ == "__main__":
    main()
