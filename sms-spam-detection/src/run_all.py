"""End-to-end pipeline: load → EDA → train → evaluate → threshold → interpret.

Run from the project root:
    python -m src.run_all
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, MODELS_DIR, RANDOM_SEED
from src.data import load_raw, deduplicate, split, spam_rate
from src.train import tune_nb, train_nb, train_lr, train_svm, cv_score, save_model
from src.evaluate import compute_metrics, pr_curve_data
from src.threshold import find_operating_points
import src.plots as P

RESULTS_FILE = FIGURES_DIR.parent / "results.json"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _get_top_words(nb_pipe, n: int = 20):
    """Extract top-n discriminative words using NB log-probability ratios."""
    vec = nb_pipe.named_steps["vec"]
    clf = nb_pipe.named_steps["clf"]
    feature_names = vec.get_feature_names_out()
    # log_prob_[class, feature]: row 1 = spam, row 0 = ham
    log_prob_spam = clf.feature_log_prob_[1]
    log_prob_ham  = clf.feature_log_prob_[0]
    ratio = log_prob_spam - log_prob_ham  # positive = more spam-like

    top_spam_idx = np.argsort(ratio)[-n:][::-1]
    top_ham_idx  = np.argsort(ratio)[:n]

    spam_words = [(feature_names[i], float(ratio[i])) for i in top_spam_idx]
    ham_words  = [(feature_names[i], float(-ratio[i])) for i in top_ham_idx]
    return spam_words, ham_words


def _error_analysis(X_test, y_test, y_prob, threshold, n: int = 5):
    """Return n false positives and n false negatives with their probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    idx_fp = np.where((y_pred == 1) & (np.array(y_test) == 0))[0]
    idx_fn = np.where((y_pred == 0) & (np.array(y_test) == 1))[0]

    texts = X_test["text"].values

    fp_examples = [
        {"text": texts[i], "prob": float(y_prob[i])}
        for i in idx_fp[:n]
    ]
    fn_examples = [
        {"text": texts[i], "prob": float(y_prob[i])}
        for i in idx_fn[:n]
    ]
    return fp_examples, fn_examples


def main():
    # ── 1. Load & clean ───────────────────────────────────────────────────────
    print("Loading data...")
    df = load_raw()
    print(f"  {len(df):,} rows, spam rate = {df['target'].mean()*100:.2f}%")

    df, n_dupes = deduplicate(df)
    print(f"  Dropped {n_dupes} duplicate messages → {len(df):,} unique rows")
    print(f"  After dedup: spam rate = {df['target'].mean()*100:.2f}%")

    # ── 2. EDA figures ─────────────────────────────────────────────────────────
    print("\nGenerating EDA figures...")
    P.class_balance(df["target"])
    P.length_distribution(df)

    # ── 3. Split ───────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = split(df)
    print(f"\nSplit: Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")
    print(f"  Spam rate  train={spam_rate(y_train)*100:.2f}%  "
          f"val={spam_rate(y_val)*100:.2f}%  test={spam_rate(y_test)*100:.2f}%")

    # ── 4. Accuracy paradox (dummy baseline) ──────────────────────────────────
    print("\nDummy baseline (always predicts ham):")
    dummy_acc = float((y_test == 0).mean())
    print(f"  Accuracy = {dummy_acc*100:.2f}%  (catches 0% spam)")
    print("  → accuracy is misleading on imbalanced data!")

    # ── 5. NB hyperparameter tuning (grid search on train CV) ─────────────────
    print("\nTuning Naive Bayes (grid search over cleaning × ngram × features × alpha)...")
    best_nb_params, best_nb_cv = tune_nb(X_train, y_train)
    print(f"  Best CV PR-AUC = {best_nb_cv:.4f}")
    print(f"  Best params: {best_nb_params}")

    # ── 6. Train all three models ──────────────────────────────────────────────
    print("\nTraining final models...")

    nb_pipe = train_nb(X_train, y_train, best_nb_params)
    nb_cv_mean, nb_cv_std = cv_score(
        # Build a fresh unfitted pipe with best params for CV
        __import__("src.features", fromlist=["nb_pipeline"]).nb_pipeline(
            cleaning=best_nb_params["cleaning"],
            vectorizer="tfidf",
            ngram_range=best_nb_params["ngram_range"],
            max_features=best_nb_params["max_features"],
            alpha=best_nb_params["alpha"],
        ),
        X_train, y_train,
    )
    nb_val_prob  = nb_pipe.predict_proba(X_val)[:, 1]
    nb_test_prob = nb_pipe.predict_proba(X_test)[:, 1]
    nb_test_m    = compute_metrics(y_test, nb_test_prob)
    print(f"  NB   CV PR-AUC = {nb_cv_mean:.4f} ± {nb_cv_std:.4f} | "
          f"Test PR-AUC = {nb_test_m['pr_auc']:.4f}")
    save_model(nb_pipe, "naive_bayes")

    lr_pipe = train_lr(X_train, y_train)
    lr_cv_mean, lr_cv_std = cv_score(
        __import__("src.features", fromlist=["lr_pipeline"]).lr_pipeline(),
        X_train, y_train,
    )
    lr_test_prob = lr_pipe.predict_proba(X_test)[:, 1]
    lr_test_m    = compute_metrics(y_test, lr_test_prob)
    print(f"  LR   CV PR-AUC = {lr_cv_mean:.4f} ± {lr_cv_std:.4f} | "
          f"Test PR-AUC = {lr_test_m['pr_auc']:.4f}")
    save_model(lr_pipe, "logistic_regression")

    svm_pipe = train_svm(X_train, y_train)
    svm_cv_mean, svm_cv_std = cv_score(
        __import__("src.features", fromlist=["svm_pipeline"]).svm_pipeline(),
        X_train, y_train,
    )
    svm_test_prob = svm_pipe.predict_proba(X_test)[:, 1]
    svm_test_m    = compute_metrics(y_test, svm_test_prob)
    print(f"  SVM  CV PR-AUC = {svm_cv_mean:.4f} ± {svm_cv_std:.4f} | "
          f"Test PR-AUC = {svm_test_m['pr_auc']:.4f}")
    save_model(svm_pipe, "linear_svm")

    # ── 7. PR curves (test set) ────────────────────────────────────────────────
    print("\nSaving PR curves...")
    nb_p, nb_r, _   = pr_curve_data(y_test, nb_test_prob)
    lr_p, lr_r, _   = pr_curve_data(y_test, lr_test_prob)
    svm_p, svm_r, _ = pr_curve_data(y_test, svm_test_prob)
    P.pr_curves({
        "Naive Bayes (tuned)":    (nb_p, nb_r, nb_test_m["pr_auc"]),
        "Logistic Regression":    (lr_p, lr_r, lr_test_m["pr_auc"]),
        "Linear SVM (calibrated)":(svm_p, svm_r, svm_test_m["pr_auc"]),
    })

    # ── 8. Threshold selection on VALIDATION set (NB) ─────────────────────────
    print("\nThreshold analysis (NB, validation set)...")
    nb_val_m = compute_metrics(y_val, nb_val_prob)
    ops, val_prec, val_rec, val_thresh = find_operating_points(y_val, nb_val_prob)
    chosen_threshold = ops["max_f1"]["threshold"]
    print(f"  Chosen threshold (max F1 on val): {chosen_threshold:.4f}")

    P.precision_recall_vs_threshold(val_prec, val_rec, val_thresh, chosen_threshold)
    P.pr_curve_with_operating_points(val_prec, val_rec, val_thresh, ops, nb_val_m["pr_auc"])

    # ── 9. Final test-set evaluation at chosen threshold ──────────────────────
    print("\nFinal test evaluation at chosen threshold:")
    final_m = compute_metrics(y_test, nb_test_prob, threshold=chosen_threshold)
    print(f"  Precision={final_m['precision']:.4f}  Recall={final_m['recall']:.4f}  "
          f"F1={final_m['f1']:.4f}  PR-AUC={final_m['pr_auc']:.4f}")
    print(f"  TP={final_m['tp']}  FP={final_m['fp']}  TN={final_m['tn']}  FN={final_m['fn']}")

    nb_pred = (nb_test_prob >= chosen_threshold).astype(int)
    P.confusion_matrix_plot(
        y_test, nb_pred,
        title=f"Naive Bayes confusion matrix (threshold = {chosen_threshold:.3f})",
    )

    # ── 10. Operating-point table ──────────────────────────────────────────────
    print("\nOperating points (thresholds on val, applied to test):")
    rows = []
    for key, op in ops.items():
        t     = op["threshold"]
        test_m = compute_metrics(y_test, nb_test_prob, threshold=t)
        rows.append({
            "Operating point":  op["name"],
            "Threshold":        f"{t:.3f}",
            "Precision (test)": f"{test_m['precision']:.4f}",
            "Recall (test)":    f"{test_m['recall']:.4f}",
            "F1 (test)":        f"{test_m['f1']:.4f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ── 11. Error analysis ─────────────────────────────────────────────────────
    print("\nError analysis (up to 5 false positives and 5 false negatives):")
    fp_examples, fn_examples = _error_analysis(
        X_test, y_test, nb_test_prob, chosen_threshold
    )
    print("  False positives (ham flagged as spam):")
    for ex in fp_examples:
        print(f"    [{ex['prob']:.3f}] {ex['text'][:80]}")
    print("  False negatives (spam that slipped through):")
    for ex in fn_examples:
        print(f"    [{ex['prob']:.3f}] {ex['text'][:80]}")

    # ── 12. Top discriminative words ──────────────────────────────────────────
    print("\nSaving top-words figure...")
    spam_words, ham_words = _get_top_words(nb_pipe, n=20)
    P.top_words(spam_words, ham_words)
    print("  Top spam words:", [w for w, _ in spam_words[:5]])
    print("  Top ham  words:", [w for w, _ in ham_words[:5]])

    # ── 13. Save chosen threshold alongside model ──────────────────────────────
    with open(MODELS_DIR / "threshold.json", "w") as f:
        json.dump({"chosen_threshold": chosen_threshold}, f)

    # ── 14. Persist all results ────────────────────────────────────────────────
    results = {
        "dataset": {
            "total_rows": len(df),
            "n_duplicates_removed": n_dupes,
            "spam_rate": float(df["target"].mean()),
        },
        "dummy": {
            "accuracy": dummy_acc,
            "recall":   0.0,
        },
        "naive_bayes": {
            "best_params":    best_nb_params,
            "cv_pr_auc_mean": nb_cv_mean,
            "cv_pr_auc_std":  nb_cv_std,
            "chosen_threshold": chosen_threshold,
            **{f"test_{k}": v for k, v in final_m.items()},
        },
        "logistic_regression": {
            "cv_pr_auc_mean": lr_cv_mean,
            "cv_pr_auc_std":  lr_cv_std,
            **{f"test_{k}": v for k, v in lr_test_m.items()},
        },
        "linear_svm": {
            "cv_pr_auc_mean": svm_cv_mean,
            "cv_pr_auc_std":  svm_cv_std,
            **{f"test_{k}": v for k, v in svm_test_m.items()},
        },
        "operating_points": {k: v for k, v in ops.items()},
        "top_spam_words": spam_words[:20],
        "top_ham_words":  ham_words[:20],
        "false_positives": fp_examples,
        "false_negatives": fn_examples,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    print("Done! All figures in reports/figures/")
    return results


if __name__ == "__main__":
    main()
