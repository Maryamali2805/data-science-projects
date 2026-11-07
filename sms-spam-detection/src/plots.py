"""Save all figures to reports/figures/."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import FIGURES_DIR, RANDOM_SEED

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
}
plt.rcParams.update(STYLE)
PALETTE = {"Ham": "#4878d0", "Spam": "#ee854a"}


def save(name: str):
    path = FIGURES_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def class_balance(y):
    """Bar + pie of ham vs spam counts."""
    spam_count = int(y.sum())
    ham_count  = int((y == 0).sum())
    total      = len(y)
    labels = ["Ham", "Spam"]
    vals   = [ham_count, spam_count]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(labels, vals, color=[PALETTE["Ham"], PALETTE["Spam"]])
    axes[0].set_title("Message counts")
    axes[0].set_ylabel("Count")
    for patch, v in zip(axes[0].patches, vals):
        axes[0].text(
            patch.get_x() + patch.get_width() / 2, v + 30,
            f"{v:,}", ha="center", va="bottom", fontsize=9
        )

    pcts = [v / total * 100 for v in vals]
    axes[1].pie(
        pcts,
        labels=[f"{l}\n{p:.1f}%" for l, p in zip(labels, pcts)],
        colors=[PALETTE["Ham"], PALETTE["Spam"]],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    axes[1].set_title("Class proportions")

    fig.suptitle(
        f"Class imbalance: {spam_count:,} spam in {total:,} messages "
        f"({spam_count/total*100:.1f}% spam)",
        fontsize=11, y=1.01,
    )
    save("class_balance")


def length_distribution(df):
    """Box + histogram of message length by class."""
    df = df.copy()
    df["length"] = df["text"].str.len()
    df["Class"]  = df["target"].map({0: "Ham", 1: "Spam"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for cls, color in [("Ham", PALETTE["Ham"]), ("Spam", PALETTE["Spam"])]:
        sub = df[df["Class"] == cls]["length"]
        axes[0].hist(sub, bins=40, alpha=0.6, label=cls, color=color, density=True)
    axes[0].set_xlabel("Message length (chars)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Message length distribution")
    axes[0].legend()

    sns.boxplot(
        data=df, x="Class", y="length",
        palette={"Ham": PALETTE["Ham"], "Spam": PALETTE["Spam"]},
        ax=axes[1],
    )
    axes[1].set_title("Length by class (box plot)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Message length (chars)")

    plt.tight_layout()
    save("length_distribution")


def top_words(spam_words: list, ham_words: list):
    """Horizontal bar charts of top discriminative words for spam and ham."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, words_scores, title, color in [
        (axes[0], spam_words, "Top spam words", PALETTE["Spam"]),
        (axes[1], ham_words,  "Top ham words",  PALETTE["Ham"]),
    ]:
        words, scores = zip(*words_scores)
        ax.barh(list(words), list(scores), color=color)
        ax.set_title(title)
        ax.set_xlabel("Log-probability ratio (spam / ham)")
        ax.invert_yaxis()

    plt.suptitle("Most discriminative words (Naive Bayes log-probability ratios)",
                 fontsize=11)
    plt.tight_layout()
    save("top_words")


def pr_curves(models_data: dict):
    """models_data: {name: (precision_arr, recall_arr, pr_auc)}"""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4878d0", "#ee854a", "#6acc65"]
    for (name, (p, r, auc)), color in zip(models_data.items(), colors):
        ax.plot(r, p, lw=2, color=color, label=f"{name} (PR-AUC = {auc:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (test set)")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    save("pr_curves")


def precision_recall_vs_threshold(precision, recall, thresholds, chosen_threshold):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precision[:-1], label="Precision", color="#4878d0", lw=2)
    ax.plot(thresholds, recall[:-1],    label="Recall",    color="#ee854a", lw=2)
    ax.axvline(
        chosen_threshold, color="grey", linestyle="--",
        label=f"Chosen threshold = {chosen_threshold:.3f}",
    )
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
            label=f"Naive Bayes (PR-AUC = {pr_auc:.4f})")

    markers = ["o", "s", "^"]
    colors  = ["#ee854a", "#6acc65", "#d65f5f"]
    for (key, pt), marker, color in zip(operating_points.items(), markers, colors):
        ax.scatter(
            pt["recall"], pt["precision"],
            marker=marker, s=120, zorder=5, color=color,
            label=f"{pt['name']} (t={pt['threshold']:.3f})",
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR curve with operating points (NB, validation set)")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    save("pr_curve_operating_points")


def confusion_matrix_plot(y_true, y_pred, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, colorbar=False,
        cmap="Blues",
        display_labels=["Ham", "Spam"],
    )
    ax.set_title(title)
    save("confusion_matrix")
