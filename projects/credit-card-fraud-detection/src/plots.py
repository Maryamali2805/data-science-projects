"""Save all figures to reports/figures/."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import FIGURES_DIR, RANDOM_SEED

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
}
plt.rcParams.update(STYLE)
PALETTE = {"Legitimate": "#4878d0", "Fraud": "#ee854a"}


def save(name: str):
    path = FIGURES_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def class_distribution(y):
    counts = y.value_counts()
    labels = ["Legitimate", "Fraud"]
    vals   = [counts[0], counts[1]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(labels, vals, color=[PALETTE["Legitimate"], PALETTE["Fraud"]])
    axes[0].set_title("Transaction counts")
    axes[0].set_ylabel("Count")
    for ax, v in zip(axes[0].patches, vals):
        axes[0].text(ax.get_x() + ax.get_width()/2, v + 500,
                     f"{v:,}", ha="center", va="bottom", fontsize=9)

    pcts = [v / sum(vals) * 100 for v in vals]
    axes[1].pie(pcts, labels=[f"{l}\n{p:.2f}%" for l, p in zip(labels, pcts)],
                colors=[PALETTE["Legitimate"], PALETTE["Fraud"]],
                startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[1].set_title("Class proportions")

    fig.suptitle("Severe class imbalance: 492 frauds in 284,807 transactions (~0.17%)",
                 fontsize=11, y=1.01)
    save("class_distribution")


def amount_time_by_class(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for cls, label, color in [(0, "Legitimate", PALETTE["Legitimate"]),
                               (1, "Fraud", PALETTE["Fraud"])]:
        sub = df[df["Class"] == cls]
        axes[0].hist(sub["Amount"], bins=60, alpha=0.6, label=label,
                     color=color, density=True)
        axes[1].scatter(sub["Time"] / 3600, sub["Amount"],
                        alpha=0.3, s=4, color=color, label=label)

    axes[0].set_title("Transaction Amount distribution")
    axes[0].set_xlabel("Amount (EUR)")
    axes[0].set_ylabel("Density")
    axes[0].set_xlim(0, 2500)
    axes[0].legend()

    axes[1].set_title("Amount vs Time by class")
    axes[1].set_xlabel("Time (hours)")
    axes[1].set_ylabel("Amount (EUR)")
    axes[1].legend(markerscale=4)

    plt.tight_layout()
    save("amount_time_by_class")


def pr_curves(models_data: dict):
    """models_data: {name: (precision_arr, recall_arr, pr_auc)}"""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4878d0", "#ee854a"]
    for (name, (p, r, auc)), color in zip(models_data.items(), colors):
        ax.plot(r, p, lw=2, color=color, label=f"{name} (PR-AUC = {auc:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (test set)")
    ax.legend()
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    save("pr_curves")


def precision_recall_vs_threshold(precision, recall, thresholds, chosen_threshold):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precision[:-1], label="Precision", color="#4878d0", lw=2)
    ax.plot(thresholds, recall[:-1], label="Recall", color="#ee854a", lw=2)
    ax.axvline(chosen_threshold, color="grey", linestyle="--",
               label=f"Chosen threshold = {chosen_threshold:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision and Recall vs decision threshold (validation set)")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.grid(alpha=0.3)
    save("precision_recall_vs_threshold")


def pr_curve_with_operating_points(precision, recall, thresholds,
                                   operating_points: dict, pr_auc: float):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, color="#4878d0",
            label=f"XGBoost (PR-AUC = {pr_auc:.4f})")

    markers = ["o", "s", "^", "D"]
    colors  = ["#ee854a", "#6acc65", "#d65f5f", "#956cb4"]
    for (key, pt), marker, color in zip(operating_points.items(), markers, colors):
        ax.scatter(pt["recall"], pt["precision"],
                   marker=marker, s=120, zorder=5, color=color,
                   label=f"{pt['name']} (t={pt['threshold']:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR curve with operating points (XGBoost, validation set)")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    save("pr_curve_operating_points")


def confusion_matrix_plot(y_true, y_pred, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, colorbar=False,
        cmap="Blues",
    )
    ax.set_title(title)
    save("confusion_matrix")


def feature_importance(model_pipeline, feature_names, top_n=20):
    clf = model_pipeline.named_steps["clf"]
    importances = clf.feature_importances_
    # Reconstruct transformed feature order: scaled cols first, then v-cols passthrough
    from src.features import SCALE_COLS
    v_cols = [c for c in feature_names if c not in SCALE_COLS]
    transformed_names = SCALE_COLS + v_cols

    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([transformed_names[i] for i in idx], importances[idx],
            color="#4878d0")
    ax.set_xlabel("Feature importance (gain)")
    ax.set_title(f"XGBoost top-{top_n} feature importances\n"
                 "(V1–V28 are anonymised PCA components)")
    plt.tight_layout()
    save("feature_importance")
