import numpy as np
import pytest

from fnews.ml.models.multimodal import gated_features, validate_alignment


def test_alignment_validation_fails_for_shuffled_ids() -> None:
    ids = np.array(["a", "b", "c"])
    labels = np.array([0, 1, 0])

    validate_alignment(ids, ids.copy(), labels, labels.copy())
    with pytest.raises(ValueError):
        validate_alignment(ids, np.array(["b", "a", "c"]), labels, labels.copy())


def test_gated_features_shape() -> None:
    text = np.ones((4, 3))
    image = np.zeros((4, 2))
    fused, gate = gated_features(text, image)

    assert fused.shape == (4, 2)
    assert gate.shape == (4, 2)
    assert np.all(gate > 0.5)
