from __future__ import annotations

from pathlib import Path

import numpy as np

from fnews.ml.preprocessing.image import load_image_rgb_array


def handcrafted_image_feature_names(histogram_bins: int = 16) -> list[str]:
    names = [
        "original_width",
        "original_height",
        "original_aspect_ratio",
        "rgb_mean_r",
        "rgb_mean_g",
        "rgb_mean_b",
        "rgb_std_r",
        "rgb_std_g",
        "rgb_std_b",
        "brightness_mean",
        "brightness_std",
        "edge_mean",
        "edge_std",
    ]
    for channel in ["r", "g", "b"]:
        for bin_index in range(histogram_bins):
            names.append(f"hist_{channel}_{bin_index}")
    return names


def extract_handcrafted_image_features(
    path: str | Path,
    *,
    image_size: tuple[int, int] = (64, 64),
    histogram_bins: int = 16,
) -> np.ndarray:
    array, original_size = load_image_rgb_array(path, image_size)
    width, height = original_size
    means = array.mean(axis=(0, 1))
    stds = array.std(axis=(0, 1))
    brightness = rgb_to_luminance(array)
    edges = sobel_edge_magnitude(brightness)
    histogram = rgb_histogram(array, histogram_bins)
    scalar_features = np.array(
        [
            width,
            height,
            width / height if height else 0.0,
            *means.tolist(),
            *stds.tolist(),
            float(brightness.mean()),
            float(brightness.std()),
            float(edges.mean()),
            float(edges.std()),
        ],
        dtype=np.float64,
    )
    return np.concatenate([scalar_features, histogram])


def rgb_histogram(array: np.ndarray, bins: int = 16) -> np.ndarray:
    pieces = []
    for channel in range(3):
        counts, _ = np.histogram(array[:, :, channel], bins=bins, range=(0.0, 1.0))
        pieces.append(counts.astype(np.float64) / array[:, :, channel].size)
    return np.concatenate(pieces)


def rgb_to_luminance(array: np.ndarray) -> np.ndarray:
    return 0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]


def sobel_edge_magnitude(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="edge")
    gx = (
        -padded[:-2, :-2]
        + padded[:-2, 2:]
        - 2 * padded[1:-1, :-2]
        + 2 * padded[1:-1, 2:]
        - padded[2:, :-2]
        + padded[2:, 2:]
    )
    gy = (
        padded[:-2, :-2]
        + 2 * padded[:-2, 1:-1]
        + padded[:-2, 2:]
        - padded[2:, :-2]
        - 2 * padded[2:, 1:-1]
        - padded[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy)


def dense_rows_to_sparse(rows: np.ndarray, epsilon: float = 0.0) -> list[dict[int, float]]:
    sparse_rows: list[dict[int, float]] = []
    for row in rows:
        sparse_rows.append({int(index): float(value) for index, value in enumerate(row) if abs(value) > epsilon})
    return sparse_rows
