"""Report figures, always drawn from the computed results (never from typed-in numbers)."""
import matplotlib

matplotlib.use("Agg")  # write PNG files; no window needed

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score, roc_curve

# Categorical slots 1-3 of a colour-blind-safe palette (validated as a set), plus quiet chart chrome.
MODEL_COLORS = {"random_forest": "#2a78d6", "lightgbm": "#eb6834", "xgboost": "#1baf7a"}
MODEL_LABELS = {"random_forest": "Random Forest", "lightgbm": "LightGBM", "xgboost": "XGBoost"}
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE_RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.edgecolor": BASELINE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "legend.labelcolor": INK_2,
})


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def _grouped_bars(ax, groups, series, width=0.26):
    """series: list of (label, color, values) drawn side by side for each group."""
    x = np.arange(len(groups))
    offset = (len(series) - 1) / 2
    for i, (label, color, values) in enumerate(series):
        ax.bar(x + (i - offset) * width, values, width, color=color, label=label, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper right", ncols=len(series))


def metric_bars(results, path):
    metrics = [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]
    series = []
    for name, res in results.items():
        test, tuned = res["test"], res["test"]["at_tuned_threshold"]
        series.append((MODEL_LABELS[name], MODEL_COLORS[name], [test.get(key, tuned.get(key)) for _, key in metrics]))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    _grouped_bars(ax, [label for label, _ in metrics], series)
    ax.set_ylabel("Score on the test set")
    ax.set_title("Test-set scores (precision, recall and F1 at each model's validation-chosen threshold)", fontsize=11)
    _save(fig, path)


def roc_curves(y_true, probs, path):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.grid(axis="both")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Random guess (0.500)")
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=2,
                label=f"{MODEL_LABELS[name]} ({roc_auc_score(y_true, prob):.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate (recall)")
    ax.set_title("ROC curves on the test set (AUC in brackets)", fontsize=11)
    ax.legend(loc="lower right")
    _save(fig, path)


def confusion_matrices(results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.9))
    for ax, (name, res) in zip(axes, results.items()):
        cm = np.array(res["test"]["at_tuned_threshold"]["confusion_matrix"])
        share = cm / cm.sum(axis=1, keepdims=True)  # colour = share of each actual class
        ax.imshow(share, cmap=BLUE_RAMP, vmin=0, vmax=1)
        ax.grid(False)
        for (i, j), count in np.ndenumerate(cm):
            ax.text(j, i, f"{count:,}\n{share[i, j]:.1%}", ha="center", va="center", fontsize=10,
                    color="white" if share[i, j] > 0.5 else INK)
        ax.set_xticks([0, 1], ["Non-default", "Default"])
        ax.set_yticks([0, 1], ["Non-default", "Default"])
        ax.set(xlabel="Predicted", ylabel="Actual")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{MODEL_LABELS[name]} (threshold {res['threshold']:.2f})", fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def leakage_bars(results, path):
    """ROC-AUC and recall for the original row-level data under the two ways of splitting it."""
    names = list(results["row_split"])
    splits = [("Row split (original, leaky)", BASELINE, "row_split"), ("Applicant split (honest)", MODEL_COLORS["random_forest"], "applicant_split")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (metric, label) in zip(axes, [("roc_auc", "ROC-AUC"), ("recall", "Recall at 0.5")]):
        series = [(split_label, color, [results[key][n][metric] for n in names]) for split_label, color, key in splits]
        _grouped_bars(ax, [MODEL_LABELS[n] for n in names], series, width=0.36)
        ax.set_title(label, fontsize=11)
    axes[1].get_legend().remove()
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncols=2)
    fig.suptitle("Same 1.5M-row data and models, two ways of splitting it", x=0.07, ha="left", fontsize=12)
    fig.tight_layout()
    _save(fig, path)
