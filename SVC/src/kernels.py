"""Kernel functions implemented with standard-library arithmetic only."""
import math

from .linalg import dot, squared_distance


def linear(left, right):
    return dot(left, right)


def polynomial(left, right, gamma=1.0, degree=3, coef0=1.0):
    if gamma <= 0 or degree < 1:
        raise ValueError("gamma must be positive and degree must be at least one")
    return (gamma * dot(left, right) + coef0) ** degree


def rbf(left, right, gamma=1.0):
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return math.exp(-gamma * squared_distance(left, right))


def resolve(name, gamma=1.0, degree=3, coef0=1.0):
    if name == "linear":
        return linear
    if name == "rbf":
        return lambda left, right: rbf(left, right, gamma)
    if name in {"poly", "polynomial"}:
        return lambda left, right: polynomial(left, right, gamma, degree, coef0)
    raise ValueError("kernel must be 'linear', 'rbf', or 'poly'")
