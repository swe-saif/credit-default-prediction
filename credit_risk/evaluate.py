"""Metrics for an imbalanced yes/no problem (about 8% of applicants default)."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def pick_threshold(y_true, prob):
    """The probability cut-off with the best F1 score, chosen on the validation set (never on the test set)."""
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])  # the curve's last point has no threshold


def score(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {
        "threshold": round(threshold, 4),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),  # [[TN, FP], [FN, TP]]
    }


def ranking_scores(y_true, prob):
    """Threshold-free scores: how well the model ranks defaulters above non-defaulters."""
    return {"roc_auc": roc_auc_score(y_true, prob), "pr_auc": average_precision_score(y_true, prob)}
