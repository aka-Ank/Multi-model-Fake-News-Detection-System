from __future__ import annotations

import numpy as np

from fnews.ml.metrics.classification import binary_classification_metrics, confusion_matrix_binary
from fnews.ml.nn.activations import sigmoid
from fnews.ml.nn.losses import binary_cross_entropy


def probability_to_logit(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def temperature_scale_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return sigmoid(probability_to_logit(probabilities) / temperature)


def find_temperature_grid(
    labels: np.ndarray,
    probabilities: np.ndarray,
    temperatures: np.ndarray | None = None,
) -> tuple[float, float]:
    if temperatures is None:
        temperatures = np.concatenate(
            [
                np.linspace(0.25, 1.0, 16),
                np.linspace(1.1, 5.0, 40),
            ]
        )
    best_temperature = float(temperatures[0])
    best_loss = float("inf")
    for temperature in temperatures:
        scaled = temperature_scale_probabilities(probabilities, float(temperature))
        loss = binary_cross_entropy(scaled, labels)
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    return best_temperature, best_loss


def reliability_bins(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []
    for index in range(bins):
        low = edges[index]
        high = edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)
        count = int(mask.sum())
        if count:
            avg_confidence = float(probabilities[mask].mean())
            empirical_accuracy = float(labels[mask].mean())
        else:
            avg_confidence = 0.0
            empirical_accuracy = 0.0
        rows.append(
            {
                "bin": float(index),
                "bin_low": float(low),
                "bin_high": float(high),
                "count": float(count),
                "avg_confidence": avg_confidence,
                "empirical_positive_rate": empirical_accuracy,
                "abs_gap": abs(empirical_accuracy - avg_confidence) if count else 0.0,
            }
        )
    return rows


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    total = len(labels)
    if total == 0:
        return 0.0
    return float(
        sum(row["count"] / total * row["abs_gap"] for row in reliability_bins(labels, probabilities, bins=bins))
    )


def threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, float]]:
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        metrics = binary_classification_metrics(labels, predictions, probabilities)
        matrix = confusion_matrix_binary(labels, predictions)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "tn": float(matrix[0, 0]),
                "fp": float(matrix[0, 1]),
                "fn": float(matrix[1, 0]),
                "tp": float(matrix[1, 1]),
            }
        )
    return rows


def abstention_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bands: np.ndarray | None = None,
) -> list[dict[str, float]]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if bands is None:
        bands = np.round(np.arange(0.00, 0.46, 0.05), 2)
    rows = []
    for band in bands:
        lower = 0.5 - float(band)
        upper = 0.5 + float(band)
        decided = (probabilities < lower) | (probabilities >= upper)
        predictions = (probabilities >= upper).astype(int)
        coverage = float(decided.mean()) if len(decided) else 0.0
        if decided.any():
            decided_metrics = binary_classification_metrics(labels[decided], predictions[decided], probabilities[decided])
            selective_accuracy = decided_metrics["accuracy"]
            selective_macro_f1 = decided_metrics["macro_f1"]
        else:
            selective_accuracy = 0.0
            selective_macro_f1 = 0.0
        rows.append(
            {
                "band": float(band),
                "lower_threshold": lower,
                "upper_threshold": upper,
                "coverage": coverage,
                "abstention_rate": 1.0 - coverage,
                "selective_accuracy": selective_accuracy,
                "selective_error_rate": 1.0 - selective_accuracy if decided.any() else 0.0,
                "selective_macro_f1": selective_macro_f1,
                "decided_count": float(decided.sum()),
                "uncertain_count": float((~decided).sum()),
            }
        )
    return rows


def bootstrap_macro_f1(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    seed: int,
    samples: int = 500,
    threshold: float = 0.5,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    values = np.zeros(samples, dtype=float)
    for index in range(samples):
        sampled = rng.integers(0, len(labels), size=len(labels))
        predictions = (probabilities[sampled] >= threshold).astype(int)
        values[index] = binary_classification_metrics(labels[sampled], predictions, probabilities[sampled])["macro_f1"]
    return values


def confidence_interval(values: np.ndarray) -> dict[str, float]:
    return {
        "ci_low": float(np.percentile(values, 2.5)),
        "median": float(np.percentile(values, 50)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


def paired_disagreement_counts(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    pred_a = (np.asarray(scores_a) >= 0.5).astype(int)
    pred_b = (np.asarray(scores_b) >= 0.5).astype(int)
    a_correct = pred_a == labels
    b_correct = pred_b == labels
    n01 = int((~a_correct & b_correct).sum())
    n10 = int((a_correct & ~b_correct).sum())
    denominator = n01 + n10
    statistic = ((abs(n01 - n10) - 1) ** 2 / denominator) if denominator > 0 else 0.0
    return {"n01_a_wrong_b_correct": float(n01), "n10_a_correct_b_wrong": float(n10), "mcnemar_chi2": float(statistic)}


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2))
    return float(np.sum(left_centered * right_centered) / denominator) if denominator else 0.0
