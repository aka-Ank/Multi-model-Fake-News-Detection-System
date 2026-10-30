"""Probability-level late fusion for independently calibrated model outputs."""

from __future__ import annotations


def weighted_fake_probability(text_fake: float, image_fake: float, alpha: float) -> float:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if not 0 <= text_fake <= 1 or not 0 <= image_fake <= 1:
        raise ValueError("component probabilities must be in [0, 1]")
    return float(alpha * text_fake + (1.0 - alpha) * image_fake)
