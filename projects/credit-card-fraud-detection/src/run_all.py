"""End-to-end pipeline: load → split → EDA → train → evaluate → threshold → plots.

Run from the project root:
    python -m src.run_all
"""
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, MODELS_DIR, RANDOM_SEED
from src.data import load_raw, split, fraud_rate
from src.train import (train_dummy, train_logistic, cv_score,
                       tune_xgb, train_xgb, save_model)
from src.evaluate import compute_metrics, pr_curve_data
from src.threshold import find_operating_points
import src.plots as P

RESULTS_FILE = FIGURES_DIR.parent / "results.json"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # ── 1. Load & split ───────────────────────────────────────────────────────
    print("Loading data...")
    df = load_raw()
    print(f"  {len(df):,} rows, fraud rate = {df['Class'].mean()*100:.4f}%")

    X_train, X_val, X_test, y_train, y_val, y_test = split(df)
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"  Fraud rate  train={fraud_rate(y_train)*100:.4f}%  "
          f"val={fraud_rate(y_val)*100:.4f}%  test={fraud_rate(y_test)*100:.4f}%")

    # ── 2. EDA figures ────────────────────────────────────────────────────────
    print("\nGenerating EDA figures...")
    P.class_distribution(df["Class"])
    P.amount_time_by_class(df)

    # ── 3. Dummy baseline (illustrate accuracy paradox) ───────────────────────
    print("\nDummy classifier (always predicts 'not fraud'):")
    dummy = train_dummy(X_train, y_train)
    dummy_proba = np.zeros(len(X_test))   # all-zero probability
    dummy_acc  = (dummy.predict(X_test) == y_test).mean()
    dummy_rec  = (dummy.predict(X_test)[y_test == 1] == 1).mean()
    print(f"  Accuracy = {dummy_acc*100:.2f}%  Recall = {dummy_rec*100:.2f}%")
    print("  → 99.83% accuracy, 0% fraud caught: accuracy is misleading!")

    # ── 4. Logistic Regression ────────────────────────────────────────────────
    print("\nTraining Logistic Regression baseline...")
    lr_pipe = train_logistic(X_train, y_train)
    lr_cv_mean, lr_cv_std = cv_score(
        logistic_pipeline_fresh(X_train.columns), X_train, y_train
    )
    lr_val_prob  = lr_pipe.predict_proba(X_val)[:, 1]
    lr_test_prob = lr_pipe.predict_proba(X_test)[:, 1]
    lr_val_m  = compute_metrics(y_val,  lr_val_prob)
    lr_test_m = compute_metrics(y_test, lr_test_prob)
    print(f"  CV PR-AUC = {lr_cv_mean:.4f} ± {lr_cv_std:.4f}")
    print(f"  Test PR-AUC = {lr_test_m['pr_auc']:.4f}  "
          f"Precision = {lr_test_m['precision']:.4f}  "
          f"Recall = {lr_test_m['recall']:.4f}  "
          f"F1 = {lr_test_m['f1']:.4f}")
    save_model(lr_pipe, "logistic_regression")

    # ── 5. XGBoost tuning ─────────────────────────────────────────────────────
    print("\nTuning XGBoost (50 Optuna trials)...")
    best_params, best_val_prauc = tune_xgb(X_train, y_train, X_val, y_val, n_trials=50)
    print(f"  Best val PR-AUC = {best_val_prauc:.4f}")
    print(f"  Best params: {best_params}")

    xgb_pipe = train_xgb(X_train, y_train, X_val, y_val, best_params)
    xgb_cv_mean, xgb_cv_std = cv_score(
        xgb_pipeline_fresh(X_train.columns, best_params), X_train, y_train
    )
    xgb_val_prob  = xgb_pipe.predict_proba(X_val)[:, 1]
    xgb_test_prob = xgb_pipe.predict_proba(X_test)[:, 1]
    xgb_val_m  = compute_metrics(y_val,  xgb_val_prob)
    xgb_test_m = compute_metrics(y_test, xgb_test_prob)
    print(f"  CV PR-AUC = {xgb_cv_mean:.4f} ± {xgb_cv_std:.4f}")
    print(f"  Test PR-AUC = {xgb_test_m['pr_auc']:.4f}  "
          f"Precision = {xgb_test_m['precision']:.4f}  "
          f"Recall = {xgb_test_m['recall']:.4f}  "
          f"F1 = {xgb_test_m['f1']:.4f}")
    save_model(xgb_pipe, "xgboost")

    # ── 6. PR curves (test set) ────────────────────────────────────────────────
    print("\nSaving PR curves...")
    lr_p, lr_r, _ = pr_curve_data(y_test, lr_test_prob)
    xgb_p, xgb_r, _ = pr_curve_data(y_test, xgb_test_prob)
    P.pr_curves({
        "Logistic Regression": (lr_p, lr_r, lr_test_m["pr_auc"]),
        "XGBoost": (xgb_p, xgb_r, xgb_test_m["pr_auc"]),
    })

    # ── 7. Threshold selection on VALIDATION set ──────────────────────────────
    print("\nThreshold analysis (on validation set)...")
    val_amounts = X_val["Amount"].values
    ops, val_prec, val_rec, val_thresh = find_operating_points(
        y_val, xgb_val_prob, amounts=val_amounts
    )
    chosen_threshold = ops["max_f1"]["threshold"]
    print(f"  Chosen threshold (max F1 on val): {chosen_threshold:.4f}")

    P.precision_recall_vs_threshold(val_prec, val_rec, val_thresh, chosen_threshold)
    P.pr_curve_with_operating_points(
        val_prec, val_rec, val_thresh, ops, xgb_val_m["pr_auc"]
    )

    # ── 8. Final test-set evaluation at chosen threshold ──────────────────────
    print("\nFinal test-set evaluation at chosen threshold:")
    final_m = compute_metrics(y_test, xgb_test_prob, threshold=chosen_threshold)
    print(f"  Precision = {final_m['precision']:.4f}  "
          f"Recall = {final_m['recall']:.4f}  "
          f"F1 = {final_m['f1']:.4f}  "
          f"PR-AUC = {final_m['pr_auc']:.4f}")
    print(f"  TP={final_m['tp']}  FP={final_m['fp']}  "
          f"TN={final_m['tn']}  FN={final_m['fn']}")

    xgb_pred = (xgb_test_prob >= chosen_threshold).astype(int)
    P.confusion_matrix_plot(
        y_test, xgb_pred,
        title=f"XGBoost confusion matrix (threshold={chosen_threshold:.3f})"
    )

    # ── 9. Feature importance ─────────────────────────────────────────────────
    print("\nSaving feature importance plot...")
    P.feature_importance(xgb_pipe, list(X_train.columns))

    # ── 10. Operating-point table ─────────────────────────────────────────────
    print("\nOperating points (thresholds chosen on val, applied to test):")
    rows = []
    for key, op in ops.items():
        t = op["threshold"]
        test_m = compute_metrics(y_test, xgb_test_prob, threshold=t)
        # legit flagged per 1000 reviewed = FP / (TP+FP) * 1000
        total_flagged = test_m["tp"] + test_m["fp"]
        legit_per_1000 = (test_m["fp"] / total_flagged * 1000) if total_flagged else 0
        rows.append({
            "Operating point": op["name"],
            "Threshold": f"{t:.3f}",
            "Precision (test)": f"{test_m['precision']:.4f}",
            "Recall (test)": f"{test_m['recall']:.4f}",
            "F1 (test)": f"{test_m['f1']:.4f}",
            "Legit/1000 reviewed": f"{legit_per_1000:.0f}",
        })
    op_df = pd.DataFrame(rows)
    print(op_df.to_string(index=False))

    # ── 11. Persist results ───────────────────────────────────────────────────
    results = {
        "dummy": {"accuracy": dummy_acc, "recall": dummy_rec},
        "logistic": {
            "cv_pr_auc_mean": lr_cv_mean,
            "cv_pr_auc_std": lr_cv_std,
            **{f"test_{k}": v for k, v in lr_test_m.items()},
        },
        "xgboost": {
            "cv_pr_auc_mean": xgb_cv_mean,
            "cv_pr_auc_std": xgb_cv_std,
            "best_params": best_params,
            "chosen_threshold": chosen_threshold,
            **{f"test_{k}": v for k, v in final_m.items()},
        },
        "operating_points": {k: v for k, v in ops.items()},
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    print("\nDone! All figures in reports/figures/")
    return results


def logistic_pipeline_fresh(columns):
    from src.features import logistic_pipeline
    return logistic_pipeline(list(columns))


def xgb_pipeline_fresh(columns, best_params):
    from src.config import RANDOM_SEED
    from src.features import xgb_pipeline
    pos_w = None  # scale_pos_weight handled inside best_params via tune_xgb
    p = {**best_params}
    return xgb_pipeline(list(columns), **p)


if __name__ == "__main__":
    main()
