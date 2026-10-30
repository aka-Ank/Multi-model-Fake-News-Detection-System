import numpy as np
import pandas as pd

from fnews.experiments.image_research_audit import duplicate_summary, split_summary


def test_duplicate_summary_counts_cross_split_overlap() -> None:
    train = pd.DataFrame({"id": ["a"], "sha256": ["same"]})
    validation = pd.DataFrame({"id": ["b"], "sha256": ["same"]})

    result = duplicate_summary(train, validation)

    assert result["cross_split_id_overlap"] == 0
    assert result["cross_split_sha256_overlap"] == 1


def test_split_summary_uses_only_verified_local_images() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "validate"],
            "valid_image": [True, False, True],
            "2_way_label": np.array([0, 1, 1]),
        }
    )

    rows = {row["split"]: row for row in split_summary(frame)}

    assert rows["train"]["valid_local_images"] == 1
    assert rows["train"]["image_coverage"] == 0.5
    assert rows["validate"]["fake_1"] == 1
