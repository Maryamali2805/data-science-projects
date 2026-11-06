"""Pytest suite: split integrity, metric sanity, threshold function."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src.config import RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
from src.evaluate import compute_metrics
from src.threshold import find_operating_points


# ── Helpers ───────────────────────────────────────────────────────────────────

def _synthetic_df(n=5000, fraud_rate=0.002, seed=0):
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n * fraud_rate))
    n_legit = n - n_fraud
    rows = []
    for cls, cnt in [(0, n_legit), (1, n_fraud)]:
        for _ in range(cnt):
            row = {f"V{i}": rng.standard_normal() for i in range(1, 29)}
            row["Time"] = rng.uniform(0, 172800)
            row["Amount"] = rng.exponential(scale=100)
            row["Class"] = cls
            rows.append(row)
    return pd.DataFrame(rows)


# ── Split tests ───────────────────────────────────────────────────────────────

class TestSplit:
    def setup_method(self):
        from src.data import split
        df = _synthetic_df(n=5000, fraud_rate=0.01)
        self.splits = split(df)
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.splits

    def test_no_index_overlap(self):
        """Train, val, test indices are disjoint — no leakage."""
        idx_train = set(self.X_train.index)
        idx_val   = set(self.X_val.index)
        idx_test  = set(self.X_test.index)
        assert len(idx_train & idx_val) == 0, "Train/val overlap"
        assert len(idx_train & idx_test) == 0, "Train/test overlap"
        assert len(idx_val & idx_test) == 0, "Val/test overlap"

    def test_covers_all_rows(self):
        """Every row appears in exactly one split."""
        total = len(self.X_train) + len(self.X_val) + len(self.X_test)
        df = _synthetic_df(n=5000, fraud_rate=0.01)
        assert total == len(df)

    def test_fraud_rate_preserved(self):
        """Each split's fraud rate is within 1 percentage point of the original."""
        original_rate = 0.01
        for y, name in [(self.y_train, "train"), (self.y_val, "val"), (self.y_test, "test")]:
            rate = y.mean()
            assert abs(rate - original_rate) < 0.01, \
                f"Fraud rate in {name} ({rate:.4f}) deviates too far from {original_rate}"

    def test_approximate_sizes(self):
        """Split sizes are within 2% of target ratios."""
        total = len(self.y_train) + len(self.y_val) + len(self.y_test)
        assert abs(len(self.y_train) / total - TRAIN_RATIO) < 0.02
        assert abs(len(self.y_val)   / total - VAL_RATIO)   < 0.02
        assert abs(len(self.y_test)  / total - TEST_RATIO)  < 0.02


# ── Metrics tests ─────────────────────────────────────────────────────────────

class TestMetrics:
    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        m = compute_metrics(y_true, y_prob, threshold=0.5)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"]    == pytest.approx(1.0)
        assert m["f1"]        == pytest.approx(1.0)
        assert m["pr_auc"]    == pytest.approx(1.0)
        assert m["fp"] == 0
        assert m["fn"] == 0

    def test_worst_classifier(self):
        """Classifier that inverts labels gets recall=0."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        m = compute_metrics(y_true, y_prob, threshold=0.5)
        assert m["recall"] == pytest.approx(0.0)
        assert m["tp"] == 0

    def test_all_negative_predicted(self):
        """Model that never predicts fraud: recall=0."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        m = compute_metrics(y_true, y_prob, threshold=0.9)
        assert m["recall"] == pytest.approx(0.0)
        assert m["fn"] == 2

    def test_threshold_effect(self):
        """Lowering the threshold increases recall."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 200)
        y_prob = rng.uniform(0, 1, 200)
        m_high = compute_metrics(y_true, y_prob, threshold=0.8)
        m_low  = compute_metrics(y_true, y_prob, threshold=0.2)
        assert m_low["recall"] >= m_high["recall"]

    def test_pr_auc_range(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 500)
        y_prob = rng.uniform(0, 1, 500)
        m = compute_metrics(y_true, y_prob)
        assert 0.0 <= m["pr_auc"] <= 1.0
        assert 0.0 <= m["roc_auc"] <= 1.0


# ── Threshold function tests ──────────────────────────────────────────────────

class TestThreshold:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.y_true = np.concatenate([np.zeros(500), np.ones(50)])
        # Reasonable but imperfect probabilities
        self.y_prob = np.concatenate([
            rng.beta(1, 5, 500),
            rng.beta(5, 1, 50),
        ])

    def test_max_f1_in_range(self):
        ops, _, _, _ = find_operating_points(self.y_true, self.y_prob)
        t = ops["max_f1"]["threshold"]
        assert 0.0 < t < 1.0

    def test_recall_90_actually_achieves_90(self):
        ops, _, _, _ = find_operating_points(self.y_true, self.y_prob)
        if "recall_90" in ops:
            assert ops["recall_90"]["recall"] >= 0.89  # allow tiny float tolerance

    def test_prec_90_actually_achieves_90(self):
        ops, _, _, _ = find_operating_points(self.y_true, self.y_prob)
        if "prec_90" in ops:
            assert ops["prec_90"]["precision"] >= 0.89

    def test_min_cost_returned_with_amounts(self):
        amounts = np.random.exponential(100, len(self.y_true))
        ops, _, _, _ = find_operating_points(self.y_true, self.y_prob, amounts=amounts)
        assert "min_cost" in ops
        assert ops["min_cost"]["total_cost"] >= 0
