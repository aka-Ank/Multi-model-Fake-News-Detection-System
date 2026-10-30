"""Explicit Fakeddit source-label semantics used by preserved checkpoints.

Fakeddit's source ``2_way_label`` is derived from its six-way taxonomy:
``1`` is the true class (six-way class 0), while ``0`` aggregates five
non-true categories. Early FNEWS artifacts retained that raw encoding but
incorrectly described it as ``0=true, 1=fake``. Checkpoints remain valid, but
their raw outputs must be translated before presentation as a fake-class score.
"""

from __future__ import annotations

FAKEDDIT_SOURCE_TRUE = 1
FAKEDDIT_SOURCE_NON_TRUE = 0


def source_label_name(label: int) -> str:
    if label == FAKEDDIT_SOURCE_TRUE:
        return "true"
    if label == FAKEDDIT_SOURCE_NON_TRUE:
        return "non_true"
    raise ValueError("Fakeddit source label must be 0 or 1")


def fake_probability_from_source_true_probability(source_true_probability: float) -> float:
    if not 0.0 <= source_true_probability <= 1.0:
        raise ValueError("source_true_probability must be in [0, 1]")
    return float(1.0 - source_true_probability)


def fake_margin_from_source_true_margin(source_true_margin: float) -> float:
    return float(-source_true_margin)


def fake_label_from_source_prediction(source_prediction: int) -> str:
    if source_prediction == FAKEDDIT_SOURCE_TRUE:
        return "true"
    if source_prediction == FAKEDDIT_SOURCE_NON_TRUE:
        return "non_true"
    raise ValueError("Fakeddit source prediction must be 0 or 1")
