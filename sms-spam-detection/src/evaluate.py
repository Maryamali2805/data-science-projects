"""Metric helpers — same interface as credit-card-fraud-detection/src/evaluate.py."""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Return a dict of all relevant metrics at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    pr_auc  = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "pr_auc":    pr_auc,
        "roc_auc":   roc_auc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "threshold": threshold,
    }


def pr_curve_data(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return precision, recall, thresholds
