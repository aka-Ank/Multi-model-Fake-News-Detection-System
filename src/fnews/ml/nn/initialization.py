from __future__ import annotations

import numpy as np


def xavier_uniform(
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return Xavier/Glorot-uniform weights for a dense layer."""
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError("fan_in and fan_out must be positive")
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64)


def he_uniform(
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return He-uniform weights for layers followed by ReLU activations."""
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError("fan_in and fan_out must be positive")
    limit = np.sqrt(6.0 / fan_in)
    return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64)
