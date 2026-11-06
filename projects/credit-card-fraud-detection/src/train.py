"""Train baseline (Logistic Regression) and improved (XGBoost) models.

Both are trained inside sklearn Pipelines to prevent leakage.
XGBoost is tuned with Optuna (PR-AUC scored on the validation set).
"""
import json
import warnings
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score
import joblib

from src.config import RANDOM_SEED, CV_FOLDS, MODELS_DIR
from src.features import logistic_pipeline, xgb_pipeline

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_dummy(X_train, y_train):
    clf = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    return clf


def train_logistic(X_train, y_train):
    pipe = logistic_pipeline(list(X_train.columns))
    pipe.fit(X_train, y_train)
    return pipe


def cv_score(pipe, X, y, scoring="average_precision"):
    """Stratified CV PR-AUC mean and std."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return scores.mean(), scores.std()


def tune_xgb(X_train, y_train, X_val, y_val, n_trials=50):
    """Use Optuna to find the best XGBoost hyperparameters (scored on val PR-AUC)."""
    pos_weight = int((y_train == 0).sum() / (y_train == 1).sum())

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0, 5),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            scale_pos_weight=pos_weight,
        )
        pipe = xgb_pipeline(list(X_train.columns), **params)
        # XGBoost early stopping needs eval_set; pass through the clf step
        fit_params = {
            "clf__eval_set": [(pipe.named_steps["pre"].fit_transform(X_val), y_val)],
            "clf__verbose": False,
        }
        pipe.named_steps["pre"].fit(X_train)
        X_tr_t = pipe.named_steps["pre"].transform(X_train)
        X_va_t = pipe.named_steps["pre"].transform(X_val)

        import xgboost as xgb_lib
        clf = xgb_lib.XGBClassifier(
            eval_metric="aucpr",
            early_stopping_rounds=20,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            **params,
        )
        clf.fit(X_tr_t, y_train,
                eval_set=[(X_va_t, y_val)],
                verbose=False)
        prob = clf.predict_proba(X_va_t)[:, 1]
        return average_precision_score(y_val, prob)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def train_xgb(X_train, y_train, X_val, y_val, best_params):
    pos_weight = int((y_train == 0).sum() / (y_train == 1).sum())
    params = {**best_params, "scale_pos_weight": pos_weight}
    pipe = xgb_pipeline(list(X_train.columns), **params)

    # Fit preprocessor first so we can pass eval_set to XGBoost
    pipe.named_steps["pre"].fit(X_train)
    X_tr_t = pipe.named_steps["pre"].transform(X_train)
    X_va_t = pipe.named_steps["pre"].transform(X_val)

    pipe.named_steps["clf"].fit(
        X_tr_t, y_train,
        eval_set=[(X_va_t, y_val)],
        verbose=False,
    )
    return pipe


def save_model(pipe, name: str):
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(pipe, path)
    print(f"  Saved model to {path}")
    return path


def load_model(name: str):
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)
