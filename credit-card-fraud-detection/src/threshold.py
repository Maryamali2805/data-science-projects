"""Threshold selection on the validation set.

All operating points are computed on val; the chosen threshold is then applied
unchanged to the test set.
"""
import numpy as np
from sklearn.metrics import precision_recall_curve

from src.config import FALSE_ALARM_COST


def find_operating_points(y_true, y_prob, amounts=None):
    """Return a list of named operating points with precision/recall/threshold."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve appends precision[-1]=1, recall[-1]=0 with no threshold
    p = precision[:-1]
    r = recall[:-1]
    t = thresholds

    points = {}

    # 1. Max F1
    f1 = 2 * p * r / (p + r + 1e-9)
    idx = np.argmax(f1)
    points["max_f1"] = {
        "name": "Max F1",
        "threshold": float(t[idx]),
        "precision": float(p[idx]),
        "recall": float(r[idx]),
        "f1": float(f1[idx]),
    }

    # 2. Recall ≥ 90 %
    mask = r >= 0.90
    if mask.any():
        idx = np.argmax(p[mask])
        pts_p = p[mask]; pts_r = r[mask]; pts_t = t[mask]
        pts_f1 = f1[mask]
        points["recall_90"] = {
            "name": "Recall ≥ 90%",
            "threshold": float(pts_t[idx]),
            "precision": float(pts_p[idx]),
            "recall": float(pts_r[idx]),
            "f1": float(pts_f1[idx]),
        }

    # 3. Precision ≥ 90 %
    mask = p >= 0.90
    if mask.any():
        idx = np.argmax(r[mask])
        pts_p = p[mask]; pts_r = r[mask]; pts_t = t[mask]
        pts_f1 = f1[mask]
        points["prec_90"] = {
            "name": "Precision ≥ 90%",
            "threshold": float(pts_t[idx]),
            "precision": float(pts_p[idx]),
            "recall": float(pts_r[idx]),
            "f1": float(pts_f1[idx]),
        }

    # 4. Min cost (requires amounts)
    if amounts is not None:
        amounts_arr = np.asarray(amounts)
        y_true_arr = np.asarray(y_true)
        costs = []
        for thresh in t:
            y_pred = (y_prob >= thresh).astype(int)
            # Missed fraud: fn * mean_fraud_amount (approximate with mean of all amounts)
            fn_mask = (y_true_arr == 1) & (y_pred == 0)
            fp_mask = (y_true_arr == 0) & (y_pred == 1)
            missed_cost = amounts_arr[fn_mask].sum()
            review_cost = fp_mask.sum() * FALSE_ALARM_COST
            costs.append(missed_cost + review_cost)
        idx = int(np.argmin(costs))
        f1_cost = 2 * p[idx] * r[idx] / (p[idx] + r[idx] + 1e-9)
        points["min_cost"] = {
            "name": f"Min cost (missed fraud = Amount, review = €{FALSE_ALARM_COST})",
            "threshold": float(t[idx]),
            "precision": float(p[idx]),
            "recall": float(r[idx]),
            "f1": float(f1_cost),
            "total_cost": float(costs[idx]),
        }

    return points, precision, recall, thresholds
