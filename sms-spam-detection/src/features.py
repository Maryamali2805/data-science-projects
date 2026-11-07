"""sklearn Pipelines for SMS spam classification.

Each pipeline consists of:
  1. TextCleaner  - a stateless FunctionTransformer that applies one of the
                    cleaning strategies from preprocess.py to the raw text column
  2. Vectorizer   - CountVectorizer or TfidfVectorizer (fit ONLY on training data)
  3. Classifier   - MultinomialNB, LogisticRegression, or CalibratedLinearSVC

The vectoriser is always fitted inside the pipeline so no test/val tokens
ever influence the vocabulary or IDF weights.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import RANDOM_SEED
from src.preprocess import CLEANERS


class TextCleaner(BaseEstimator, TransformerMixin):
    """Stateless transformer that applies a named cleaning function."""

    def __init__(self, strategy: str = "minimal"):
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        clean_fn = CLEANERS[self.strategy]
        # Accept a DataFrame column, Series, or list
        if hasattr(X, "iloc"):
            texts = X.iloc[:, 0] if X.ndim == 2 else X
            return texts.map(clean_fn)
        return [clean_fn(t) for t in X]


def nb_pipeline(
    cleaning: str = "minimal",
    vectorizer: str = "tfidf",
    ngram_range: tuple = (1, 2),
    max_features: int | None = 20_000,
    min_df: int = 2,
    alpha: float = 0.1,
) -> Pipeline:
    vec = _make_vectorizer(vectorizer, ngram_range, max_features, min_df)
    return Pipeline([
        ("cleaner", TextCleaner(strategy=cleaning)),
        ("vec",     vec),
        ("clf",     MultinomialNB(alpha=alpha)),
    ])


def lr_pipeline(
    cleaning: str = "minimal",
    ngram_range: tuple = (1, 2),
    max_features: int | None = 20_000,
    min_df: int = 2,
    C: float = 1.0,
) -> Pipeline:
    vec = _make_vectorizer("tfidf", ngram_range, max_features, min_df)
    return Pipeline([
        ("cleaner", TextCleaner(strategy=cleaning)),
        ("vec",     vec),
        ("clf",     LogisticRegression(
            C=C, class_weight="balanced",
            max_iter=1000, random_state=RANDOM_SEED,
            solver="lbfgs",
        )),
    ])


def svm_pipeline(
    cleaning: str = "minimal",
    ngram_range: tuple = (1, 2),
    max_features: int | None = 20_000,
    min_df: int = 2,
    C: float = 1.0,
) -> Pipeline:
    vec = _make_vectorizer("tfidf", ngram_range, max_features, min_df)
    svc = CalibratedClassifierCV(
        LinearSVC(C=C, class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED),
        method="isotonic",
        cv=5,
    )
    return Pipeline([
        ("cleaner", TextCleaner(strategy=cleaning)),
        ("vec",     vec),
        ("clf",     svc),
    ])


def _make_vectorizer(kind, ngram_range, max_features, min_df):
    kwargs = dict(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        strip_accents="unicode",
        decode_error="replace",
        sublinear_tf=(kind == "tfidf"),
    )
    if kind == "tfidf":
        return TfidfVectorizer(**kwargs)
    return CountVectorizer(**kwargs)
