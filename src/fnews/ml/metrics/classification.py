from __future__ import annotations

import numpy as np


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    counts = confusion_counts(y_true, y_pred)
    return np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]], dtype=int)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    counts = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    total = tp + tn + fp + fn
    precision_pos = safe_divide(tp, tp + fp)
    recall_pos = safe_divide(tp, tp + fn)
    f1_pos = safe_divide(2 * precision_pos * recall_pos, precision_pos + recall_pos)
    precision_neg = safe_divide(tn, tn + fn)
    recall_neg = safe_divide(tn, tn + fp)
    f1_neg = safe_divide(2 * precision_neg * recall_neg, precision_neg + recall_neg)
    support_pos = tp + fn
    support_neg = tn + fp
    metrics = {
        "accuracy": safe_divide(tp + tn, total),
        "precision": precision_pos,
        "recall": recall_pos,
        "specificity": recall_neg,
        "f1": f1_pos,
        "macro_precision": (precision_pos + precision_neg) / 2,
        "macro_recall": (recall_pos + recall_neg) / 2,
        "macro_f1": (f1_pos + f1_neg) / 2,
        "weighted_f1": safe_divide(f1_pos * support_pos + f1_neg * support_neg, total),
        **{key: float(value) for key, value in counts.items()},
    }
    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score_manual(y_true, y_score)
        metrics["brier_score"] = brier_score(y_true, y_score)
    return metrics


def roc_curve_manual(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    order = np.argsort(-y_score)
    sorted_true = y_true[order]
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    last_score = None
    for label, score in zip(sorted_true, y_score[order], strict=True):
        if last_score is not None and score != last_score:
            tpr.append(safe_divide(tp, positives))
            fpr.append(safe_divide(fp, negatives))
        if label == 1:
            tp += 1
        else:
            fp += 1
        last_score = score
    tpr.append(safe_divide(tp, positives))
    fpr.append(safe_divide(fp, negatives))
    return np.array(fpr, dtype=float), np.array(tpr, dtype=float)


def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr = roc_curve_manual(y_true, y_score)
    area = 0.0
    for index in range(1, len(fpr)):
        width = fpr[index] - fpr[index - 1]
        height = (tpr[index] + tpr[index - 1]) / 2
        area += width * height
    return float(area)


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)
    return float(np.mean((y_score - y_true) ** 2))
