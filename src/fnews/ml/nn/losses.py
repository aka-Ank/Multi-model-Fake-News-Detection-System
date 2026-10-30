from __future__ import annotations

import numpy as np


def binary_cross_entropy(probs: np.ndarray, labels: np.ndarray, epsilon: float = 1e-12) -> float:
    probs = np.clip(np.asarray(probs, dtype=np.float64).reshape(-1), epsilon, 1 - epsilon)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    losses = -(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    return float(losses.mean())


def binary_cross_entropy_backward(probs: np.ndarray, labels: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=np.float64), epsilon, 1 - epsilon)
    labels = np.asarray(labels, dtype=np.float64).reshape(probs.shape)
    return (probs - labels) / (probs * (1 - probs) * labels.size)


def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    diff = np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64)
    return float(np.mean(diff * diff))


def mse_backward(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    return 2 * (predictions - targets) / predictions.size
