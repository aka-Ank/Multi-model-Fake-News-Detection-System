
import pandas as pd
from PIL import Image

from fnews.data.resumable_image_acquisition import (
    SUCCESS,
    DownloadResult,
    already_completed,
    initialize_database,
    record_result,
    select_training_candidates,
    validate_and_store,
)


def test_candidate_selection_is_train_only_balanced_and_resumable() -> None:
    frame = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "v"],
            "split": ["train", "train", "train", "train", "validate"],
            "image_url": ["https://example.com/a.jpg"] * 5,
            "downloaded": [False] * 5,
            "2_way_label": [0, 0, 1, 1, 1],
        }
    )

    selected = select_training_candidates(frame, {"a"}, {"b": 3}, 4, 3, seed=42)

    assert set(selected["id"]) == {"c", "d"}
    assert set(selected["split"]) == {"train"}


def test_validated_image_is_stored_and_registered(tmp_path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (3, 3), color="red").save(source)
    path, digest = validate_and_store(source.read_bytes(), "sample", "https://example.com/sample.png", tmp_path)
    assert path.exists()
    assert len(digest) == 64

    connection = initialize_database(tmp_path / "state.sqlite")
    record_result(connection, DownloadResult("sample", "https://example.com/sample.png", 1, SUCCESS, str(path), digest, 10, None))
    connection.commit()
    assert already_completed(connection) == {"sample"}
    connection.close()
