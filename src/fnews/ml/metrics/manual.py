"""Small dependency-free classification, calibration, and split helpers."""

from __future__ import annotations

import numpy as np

from fnews.ml.metrics.classification import binary_classification_metrics, confusion_matrix_binary
from fnews.ml.models.logistic_regression import sigmoid_stable


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true, dtype=int) == np.asarray(y_pred, dtype=int)))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(binary_classification_metrics(y_true, y_pred)["macro_f1"])


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix_binary(y_true, y_pred)
    result: dict[str, object] = {}
    f1s, supports = [], []
    for label, name in enumerate(("true", "fake")):
        tp, fp, fn = matrix[label, label], matrix[1 - label, label], matrix[label, 1 - label]
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        support = int(matrix[label].sum())
        result[name] = {"precision": precision, "recall": recall, "f1-score": f1, "support": support}
        f1s.append(f1)
        supports.append(support)
    total = sum(supports)
    result["accuracy"] = accuracy(y_true, y_pred)
    result["macro avg"] = {"precision": float(np.mean([result[x]["precision"] for x in ("true", "fake")])), "recall": float(np.mean([result[x]["recall"] for x in ("true", "fake")])), "f1-score": float(np.mean(f1s)), "support": total}
    result["weighted avg"] = {"precision": float(sum(result[x]["precision"] * s for x, s in zip(("true", "fake"), supports, strict=True)) / total) if total else 0.0, "recall": float(sum(result[x]["recall"] * s for x, s in zip(("true", "fake"), supports, strict=True)) / total) if total else 0.0, "f1-score": float(np.average(f1s, weights=supports)) if total else 0.0, "support": total}
    return result


def binary_log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    truth = np.asarray(y, dtype=float)
    return float(-np.mean(truth * np.log(p) + (1 - truth) * np.log(1 - p)))


class PlattCalibrator:
    """One-dimensional logistic calibration trained with batch gradient descent."""
    def __init__(self, learning_rate: float = 0.05, l2: float = 1e-6, epochs: int = 2000) -> None:
        self.learning_rate, self.l2, self.epochs = learning_rate, l2, epochs
        self.weight, self.bias = 0.0, 0.0

    def fit(self, margins: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
        x, y = np.asarray(margins, dtype=float), np.asarray(labels, dtype=float)
        for _ in range(self.epochs):
            p = sigmoid_stable(self.weight * x + self.bias)
            error = p - y
            self.weight -= self.learning_rate * (float(np.mean(error * x)) + self.l2 * self.weight)
            self.bias -= self.learning_rate * float(np.mean(error))
        return self

    def predict_proba(self, margins: np.ndarray) -> np.ndarray:
        p = sigmoid_stable(self.weight * np.asarray(margins, dtype=float) + self.bias)
        return np.c_[1 - p, p]

    def save(self, path) -> None:
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps({"weight": self.weight, "bias": self.bias}), encoding="utf-8")

    @classmethod
    def load(cls, path) -> PlattCalibrator:
        import json
        from pathlib import Path
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        item = cls()
        item.weight, item.bias = float(payload["weight"]), float(payload["bias"])
        return item


def stratified_kfold_indices(labels: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    buckets = [np.array_split(rng.permutation(np.flatnonzero(labels == value)), folds) for value in (0, 1)]
    all_indices = np.arange(len(labels))
    result = []
    for index in range(folds):
        holdout = np.concatenate((buckets[0][index], buckets[1][index]))
        holdout.sort()
        result.append((all_indices[~np.isin(all_indices, holdout)], holdout))
    return result


def stratified_half_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return stratified_kfold_indices(labels, 2, seed)[0]
