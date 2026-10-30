import pytest

from fnews.ml.label_semantics import (
    fake_label_from_source_prediction,
    fake_margin_from_source_true_margin,
    fake_probability_from_source_true_probability,
    source_label_name,
)
from fnews.ml.models.weighted_fusion import weighted_fake_probability


def test_fakeddit_raw_label_semantics_match_source_taxonomy() -> None:
    assert source_label_name(1) == "true"
    assert source_label_name(0) == "non_true"
    assert fake_label_from_source_prediction(1) == "true"
    assert fake_label_from_source_prediction(0) == "non_true"


def test_source_true_scores_translate_to_fake_scores() -> None:
    assert fake_margin_from_source_true_margin(2.5) == -2.5
    assert fake_probability_from_source_true_probability(0.85) == pytest.approx(0.15)


def test_weighted_fusion_combines_probabilities_with_the_same_non_true_semantics() -> None:
    text_non_true = fake_probability_from_source_true_probability(0.8)
    image_non_true = fake_probability_from_source_true_probability(0.3)
    assert weighted_fake_probability(text_non_true, image_non_true, 0.4) == pytest.approx(0.5)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_probability_translation_rejects_invalid_probability(value: float) -> None:
    with pytest.raises(ValueError):
        fake_probability_from_source_true_probability(value)
