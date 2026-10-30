from __future__ import annotations

import numpy as np


def relative_error(analytical: np.ndarray, numerical: np.ndarray, epsilon: float = 1e-12) -> float:
    numerator = np.linalg.norm(analytical - numerical)
    denominator = np.linalg.norm(analytical) + np.linalg.norm(numerical) + epsilon
    return float(numerator / denominator)


def numerical_gradient(
    loss_fn,
    param: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    grad = np.zeros_like(param, dtype=np.float64)
    iterator = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
    while not iterator.finished:
        index = iterator.multi_index
        original = float(param[index])
        param[index] = original + epsilon
        plus = loss_fn()
        param[index] = original - epsilon
        minus = loss_fn()
        param[index] = original
        grad[index] = (plus - minus) / (2 * epsilon)
        iterator.iternext()
    return grad
