from __future__ import annotations

import numpy as np


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, values)


def relu_backward(inputs: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    return grad_output * (inputs > 0)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1 + exp_values)
    return output


def sigmoid_backward(outputs: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    return grad_output * outputs * (1 - outputs)


def tanh(values: np.ndarray) -> np.ndarray:
    return np.tanh(values)


def tanh_backward(outputs: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    return grad_output * (1 - outputs * outputs)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)
