"""Build sklearn Pipelines that handle all feature engineering.

Scaling is fitted ONLY on training data inside each pipeline, preventing leakage.
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from src.config import RANDOM_SEED

# Columns to scale (Amount and Time are raw; V1-V28 are already PCA-normalised
# but we scale them too for LR convergence).
SCALE_COLS = ["Amount", "Time"]


def _preprocessor(feature_names):
    """Return a ColumnTransformer that scales Amount and Time, passes others through."""
    v_cols = [c for c in feature_names if c not in SCALE_COLS]
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), SCALE_COLS),
            ("passthrough", "passthrough", v_cols),
        ],
        remainder="drop",
    )


def logistic_pipeline(feature_names, C=0.01):
    pre = _preprocessor(feature_names)
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            class_weight="balanced",
            C=C,
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )),
    ])


def xgb_pipeline(feature_names, **xgb_params):
    pre = _preprocessor(feature_names)
    defaults = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=577,   # rough class ratio; will be overridden by tuning
        eval_metric="aucpr",
        # early_stopping_rounds omitted: requires eval_set at fit time, breaks CV
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    defaults.update(xgb_params)
    return Pipeline([
        ("pre", pre),
        ("clf", xgb.XGBClassifier(**defaults)),
    ])
