from __future__ import annotations

import numpy as np

from fnews.ml.metrics.calibration import (
    abstention_metrics,
    bootstrap_macro_f1,
    expected_calibration_error,
    find_temperature_grid,
    paired_disagreement_counts,
    pearson_correlation,
    reliability_bins,
    temperature_scale_probabilities,
    threshold_metrics,
)


def test_reliability_bins_and_ece_known_example() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    bins = reliability_bins(labels, scores, bins=2)

    assert bins[0]["count"] == 2
    assert np.isclose(bins[0]["avg_confidence"], 0.15)
    assert bins[0]["empirical_positive_rate"] == 0.0
    assert bins[1]["count"] == 2
    assert np.isclose(bins[1]["avg_confidence"], 0.85)
    assert bins[1]["empirical_positive_rate"] == 1.0
    assert np.isclose(expected_calibration_error(labels, scores, bins=2), 0.15)


def test_temperature_scaling_preserves_order_and_grid_finds_candidate() -> None:
    labels = np.array([0, 1, 1])
    scores = np.array([0.2, 0.6, 0.9])
    scaled = temperature_scale_probabilities(scores, 2.0)
    temperature, loss = find_temperature_grid(labels, scores, np.array([0.5, 1.0, 2.0]))

    assert list(np.argsort(scaled)) == list(np.argsort(scores))
    assert temperature in {0.5, 1.0, 2.0}
    assert loss >= 0.0


def test_threshold_metrics_and_abstention_logic() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])

    threshold_row = threshold_metrics(labels, scores, thresholds=np.array([0.5]))[0]
    abstention_row = abstention_metrics(labels, scores, bands=np.array([0.2]))[0]

    assert threshold_row["accuracy"] == 1.0
    assert threshold_row["tn"] == 2
    assert abstention_row["coverage"] == 0.5
    assert abstention_row["selective_accuracy"] == 1.0


def test_bootstrap_paired_and_correlation_outputs() -> None:
    labels = np.array([0, 0, 1, 1])
    text_scores = np.array([0.2, 0.7, 0.8, 0.4])
    fusion_scores = np.array([0.2, 0.3, 0.8, 0.9])

    boot = bootstrap_macro_f1(labels, fusion_scores, seed=7, samples=10)
    paired = paired_disagreement_counts(labels, text_scores, fusion_scores)
    corr = pearson_correlation(text_scores, fusion_scores)

    assert boot.shape == (10,)
    assert paired["n01_a_wrong_b_correct"] == 2
    assert paired["n10_a_correct_b_wrong"] == 0
    assert -1.0 <= corr <= 1.0
