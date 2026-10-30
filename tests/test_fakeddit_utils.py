import pandas as pd

from fnews.data.fakeddit import infer_split, normalize_bool_series


def test_infer_split_from_common_fakeddit_names() -> None:
    patterns = {
        "train": ("train",),
        "validate": ("validate", "val", "dev"),
        "test": ("test",),
    }

    assert infer_split("data/raw/fakeddit/all_train.tsv", patterns) == "train"
    assert infer_split("data/raw/fakeddit/all_validate.tsv", patterns) == "validate"
    assert infer_split("data/raw/fakeddit/all_test_public.tsv", patterns) == "test"
    assert infer_split("data/raw/fakeddit/comments.tsv", patterns) == "unknown"


def test_normalize_bool_series_handles_string_values() -> None:
    values = pd.Series([True, False, "true", "False", "1", "0", "yes", "", None])

    assert normalize_bool_series(values).tolist() == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        False,
    ]
