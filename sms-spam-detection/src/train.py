"""Train and compare NB, LR, and SVM classifiers.

NB is the main model; LR and SVM serve as comparison baselines.
All use sklearn Pipelines (see features.py) to prevent leakage.
Hyperparameter search for NB uses GridSearchCV on the training set.
"""
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from src.config import CV_FOLDS, MODELS_DIR, RANDOM_SEED
from src.features import lr_pipeline, nb_pipeline, svm_pipeline

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def cv_score(pipe, X_train, y_train, scoring="average_precision") -> tuple[float, float]:
    """Stratified CV PR-AUC mean and std on the training set."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def tune_nb(X_train, y_train) -> tuple[dict, float]:
    """Grid search over NB hyperparameters, scored on CV PR-AUC."""
    param_grid = {
        "cleaner__strategy": ["minimal", "heavy"],
        "vec__ngram_range":  [(1, 1), (1, 2)],
        "vec__max_features": [10_000, 30_000, None],
        "clf__alpha":        [0.01, 0.1, 1.0],
    }
    base_pipe = nb_pipeline()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    gs = GridSearchCV(
        base_pipe, param_grid,
        scoring="average_precision",
        cv=cv, n_jobs=-1, refit=True, verbose=0,
    )
    gs.fit(X_train, y_train)
    best = {
        "cleaning":     gs.best_params_["cleaner__strategy"],
        "ngram_range":  gs.best_params_["vec__ngram_range"],
        "max_features": gs.best_params_["vec__max_features"],
        "alpha":        gs.best_params_["clf__alpha"],
    }
    return best, float(gs.best_score_)


def train_nb(X_train, y_train, best_params: dict):
    """Train the final NB model with the best hyperparameters."""
    pipe = nb_pipeline(
        cleaning=best_params["cleaning"],
        vectorizer="tfidf",
        ngram_range=best_params["ngram_range"],
        max_features=best_params["max_features"],
        alpha=best_params["alpha"],
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_lr(X_train, y_train) -> object:
    pipe = lr_pipeline()
    pipe.fit(X_train, y_train)
    return pipe


def train_svm(X_train, y_train) -> object:
    pipe = svm_pipeline()
    pipe.fit(X_train, y_train)
    return pipe


def save_model(pipe, name: str):
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(pipe, path)
    print(f"  Saved model to {path}")
    return path


def load_model(name: str):
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)
