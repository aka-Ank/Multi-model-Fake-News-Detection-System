from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image_rgb_array(
    path: str | Path,
    image_size: tuple[int, int] = (64, 64),
) -> tuple[np.ndarray, tuple[int, int]]:
    """Load an image as a resized RGB float array in the [0, 1] range."""
    with Image.open(path) as image:
        original_size = image.size
        resized = image.convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float64) / 255.0
    return array, original_size
