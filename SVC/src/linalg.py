"""Small list-based linear algebra utilities used by the SVC implementations."""
from math import sqrt


def validate_vector(vector, name="vector"):
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError(f"{name} must be a non-empty list or tuple")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in vector):
        raise TypeError(f"{name} must contain only numeric values")


def validate_matrix(matrix, name="matrix"):
    if not isinstance(matrix, (list, tuple)) or not matrix:
        raise ValueError(f"{name} must be a non-empty list of rows")
    width = len(matrix[0])
    if width == 0:
        raise ValueError(f"{name} rows must be non-empty")
    for index, row in enumerate(matrix):
        validate_vector(row, f"{name}[{index}]")
        if len(row) != width:
            raise ValueError(f"{name} rows must have one shared width")


def dot(left, right):
    validate_vector(left, "left")
    validate_vector(right, "right")
    if len(left) != len(right):
        raise ValueError("dot-product vectors must have equal length")
    return sum(a * b for a, b in zip(left, right))


def add(left, right):
    if len(left) != len(right):
        raise ValueError("vectors must have equal length")
    return [a + b for a, b in zip(left, right)]


def subtract(left, right):
    if len(left) != len(right):
        raise ValueError("vectors must have equal length")
    return [a - b for a, b in zip(left, right)]


def scale(vector, scalar):
    validate_vector(vector)
    if not isinstance(scalar, (int, float)):
        raise TypeError("scalar must be numeric")
    return [scalar * value for value in vector]


def norm(vector):
    return sqrt(dot(vector, vector))


def squared_distance(left, right):
    if len(left) != len(right):
        raise ValueError("vectors must have equal length")
    return sum((a - b) ** 2 for a, b in zip(left, right))


def matvec(matrix, vector):
    validate_matrix(matrix)
    if len(matrix[0]) != len(vector):
        raise ValueError("matrix width must equal vector length")
    return [dot(row, vector) for row in matrix]
