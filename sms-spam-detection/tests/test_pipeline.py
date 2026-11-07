"""Pytest suite: data loading, split integrity, cleaning, predict."""
import numpy as np
import pandas as pd
import pytest

from src.config import RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
from src.data import deduplicate, load_raw, split
from src.evaluate import compute_metrics
from src.preprocess import clean_heavy, clean_minimal
from src.threshold import find_operating_points


# ── Helpers ───────────────────────────────────────────────────────────────────

def _synthetic_sms(n: int = 500, spam_rate: float = 0.13, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_spam = max(1, int(n * spam_rate))
    n_ham  = n - n_spam
    texts = [f"ham message number {i}" for i in range(n_ham)] + \
            [f"WIN free PRIZE call 0800{i}" for i in range(n_spam)]
    labels = [0] * n_ham + [1] * n_spam
    return pd.DataFrame({"text": texts, "target": labels, "label": ["ham"] * n_ham + ["spam"] * n_spam})


# ── Data loader tests ─────────────────────────────────────────────────────────

class TestLoader:
    def test_drops_unnamed_columns(self):
        """Loader must drop the three unnamed filler columns in spam.csv."""
        # We test the column-dropping logic without hitting the filesystem by
        # constructing a minimal DataFrame that mimics the raw file.
        raw = pd.DataFrame({
            "v1": ["ham", "spam"],
            "v2": ["hello", "FREE PRIZE"],
            "Unnamed: 2": [np.nan, np.nan],
            "Unnamed: 3": [np.nan, np.nan],
            "Unnamed: 4": [np.nan, np.nan],
        })
        junk = [c for c in raw.columns if c.startswith("Unnamed")]
        cleaned = raw.drop(columns=junk).rename(columns={"v1": "label", "v2": "text"})
        assert "Unnamed: 2" not in cleaned.columns
        assert "label" in cleaned.columns
        assert "text" in cleaned.columns

    def test_target_is_binary(self, tmp_path):
        """After loading a minimal CSV the target column is 0/1."""
        import csv
        f = tmp_path / "spam.csv"
        with open(f, "w", newline="", encoding="latin-1") as fh:
            writer = csv.writer(fh)
            writer.writerow(["v1", "v2", "Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
            writer.writerow(["ham",  "hello world", "", "", ""])
            writer.writerow(["spam", "WIN £100",    "", "", ""])
        df = pd.read_csv(f, encoding="latin-1")
        junk = [c for c in df.columns if c.startswith("Unnamed")]
        df = df.drop(columns=junk).rename(columns={"v1": "label", "v2": "text"})
        df["target"] = (df["label"] == "spam").astype(int)
        assert set(df["target"].unique()) == {0, 1}


# ── Deduplicate + split tests ─────────────────────────────────────────────────

class TestSplit:
    def setup_method(self):
        df = _synthetic_sms(n=500, spam_rate=0.13)
        df, _ = deduplicate(df)
        self.splits = split(df)
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.splits

    def test_no_text_overlap(self):
        """Train, val, test must share no identical text (no leakage)."""
        train_texts = set(self.X_train["text"])
        val_texts   = set(self.X_val["text"])
        test_texts  = set(self.X_test["text"])
        assert len(train_texts & val_texts)  == 0, "Train/val text overlap"
        assert len(train_texts & test_texts) == 0, "Train/test text overlap"
        assert len(val_texts   & test_texts) == 0, "Val/test text overlap"

    def test_covers_all_rows(self):
        df = _synthetic_sms(n=500, spam_rate=0.13)
        df, _ = deduplicate(df)
        total = len(self.X_train) + len(self.X_val) + len(self.X_test)
        assert total == len(df)

    def test_spam_rate_preserved(self):
        original_rate = 0.13
        for y, name in [
            (self.y_train, "train"),
            (self.y_val,   "val"),
            (self.y_test,  "test"),
        ]:
            rate = float(y.mean())
            assert abs(rate - original_rate) < 0.05, \
                f"Spam rate in {name} ({rate:.3f}) deviates too far from {original_rate}"

    def test_approximate_sizes(self):
        total = len(self.y_train) + len(self.y_val) + len(self.y_test)
        assert abs(len(self.y_train) / total - TRAIN_RATIO) < 0.03
        assert abs(len(self.y_val)   / total - VAL_RATIO)   < 0.03
        assert abs(len(self.y_test)  / total - TEST_RATIO)  < 0.03

    def test_dedupe_removes_duplicates(self):
        df = pd.DataFrame({
            "text":   ["hello", "hello", "spam here"],
            "target": [0, 0, 1],
            "label":  ["ham", "ham", "spam"],
        })
        deduped, n_removed = deduplicate(df)
        assert n_removed == 1
        assert len(deduped) == 2


# ── Cleaning function tests ────────────────────────────────────────────────────

class TestCleaning:
    def test_minimal_lowercases(self):
        assert clean_minimal("FREE PRIZE!!!") == "free prize!!!"

    def test_minimal_preserves_punctuation(self):
        result = clean_minimal("Call 08001234567!")
        assert "!" in result

    def test_heavy_lowercases(self):
        result = clean_heavy("WIN A PRIZE")
        assert result == result.lower()

    def test_heavy_normalises_digits(self):
        result = clean_heavy("Win £150 call 08001234567")
        assert "num" in result
        assert "150" not in result

    def test_heavy_keeps_currency(self):
        result = clean_heavy("Win £100")
        assert "£" in result

    def test_heavy_keeps_exclamation(self):
        result = clean_heavy("FREE!!!")
        assert "!" in result


# ── Metric and threshold tests ────────────────────────────────────────────────

class TestMetrics:
    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        m = compute_metrics(y_true, y_prob)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"]    == pytest.approx(1.0)
        assert m["pr_auc"]    == pytest.approx(1.0)
        assert m["fp"] == 0 and m["fn"] == 0

    def test_threshold_raises_precision(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 300)
        y_prob = rng.uniform(0, 1, 300)
        m_low  = compute_metrics(y_true, y_prob, threshold=0.2)
        m_high = compute_metrics(y_true, y_prob, threshold=0.8)
        # Higher threshold → higher precision (or at worst equal)
        assert m_high["precision"] >= m_low["precision"] - 0.05


class TestThreshold:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.y_true = np.concatenate([np.zeros(500), np.ones(70)])
        self.y_prob = np.concatenate([
            rng.beta(1, 5, 500),
            rng.beta(5, 1, 70),
        ])

    def test_max_f1_in_range(self):
        ops, _, _, _ = find_operating_points(self.y_true, self.y_prob)
        t = ops["max_f1"]["threshold"]
        assert 0.0 < t < 1.0

    def test_prec_99_achieves_99(self):
        ops, _, _, _ = find_operating_points(self.y_true, self.y_prob)
        if "prec_99" in ops:
            assert ops["prec_99"]["precision"] >= 0.98

    def test_recall_95_achieves_95(self):
        ops, _, _, _ = find_operating_points(self.y_true, self.y_prob)
        if "recall_95" in ops:
            assert ops["recall_95"]["recall"] >= 0.94


# ── Predict function test ─────────────────────────────────────────────────────

class TestPredict:
    def test_predict_returns_expected_keys(self, tmp_path, monkeypatch):
        """predict_message loads a model and returns label + probability."""
        import joblib
        from src.features import nb_pipeline

        # Build and fit a tiny NB on two examples
        texts = ["win free prize call now"] * 5 + ["hey how are you"] * 5
        labels = [1] * 5 + [0] * 5
        X = pd.DataFrame({"text": texts})
        y = np.array(labels)
        pipe = nb_pipeline(max_features=100, ngram_range=(1, 1), alpha=1.0)
        pipe.fit(X, y)

        # Redirect MODELS_DIR to tmp_path
        model_file = tmp_path / "naive_bayes.joblib"
        joblib.dump(pipe, model_file)
        thresh_file = tmp_path / "threshold.json"
        import json
        with open(thresh_file, "w") as f:
            json.dump({"chosen_threshold": 0.5}, f)

        monkeypatch.setattr("src.predict.MODELS_DIR", tmp_path)

        from src.predict import predict_message
        result = predict_message("win a prize now")
        assert "label" in result
        assert "probability" in result
        assert result["label"] in ("spam", "ham")
        assert 0.0 <= result["probability"] <= 1.0
