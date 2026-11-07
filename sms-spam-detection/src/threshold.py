"""Operating-point analysis for spam detection.

For spam, the cost asymmetry is different from fraud:
  - False positive (real message wrongly sent to spam) = user loses a message they wanted
  - False negative (spam gets through) = minor annoyance

So high precision is usually preferred over high recall in spam filters.

Operating points:
  max_f1       — balanced F1 (practical default)
  prec_99      — precision ≥ 99%  (never lose a real message)
  recall_95    — recall ≥ 95%     (catch almost all spam)
"""
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score


def find_operating_points(y_true, y_prob) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Find key operating points on the validation set.

    Returns (operating_points_dict, precision_arr, recall_arr, threshold_arr).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns arrays of len n+1; thresholds has len n
    p = precision[:-1]
    r = recall[:-1]
    t = thresholds

    # Max F1
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    best_f1_idx = int(np.argmax(f1))
    ops = {
        "max_f1": {
            "name":      "Max F1",
            "threshold": float(t[best_f1_idx]),
            "precision": float(p[best_f1_idx]),
            "recall":    float(r[best_f1_idx]),
            "f1":        float(f1[best_f1_idx]),
        }
    }

    # Precision >= 99%: highest recall among thresholds where precision >= 0.99
    mask_p99 = p >= 0.99
    if mask_p99.any():
        idx = int(np.argmax(r[mask_p99]))
        actual_idx = np.where(mask_p99)[0][idx]
        ops["prec_99"] = {
            "name":      "Precision ≥ 99%",
            "threshold": float(t[actual_idx]),
            "precision": float(p[actual_idx]),
            "recall":    float(r[actual_idx]),
            "f1":        float(f1[actual_idx]),
        }

    # Recall >= 95%: highest precision among thresholds where recall >= 0.95
    mask_r95 = r >= 0.95
    if mask_r95.any():
        idx = int(np.argmax(p[mask_r95]))
        actual_idx = np.where(mask_r95)[0][idx]
        ops["recall_95"] = {
            "name":      "Recall ≥ 95%",
            "threshold": float(t[actual_idx]),
            "precision": float(p[actual_idx]),
            "recall":    float(r[actual_idx]),
            "f1":        float(f1[actual_idx]),
        }

    return ops, precision, recall, thresholds
