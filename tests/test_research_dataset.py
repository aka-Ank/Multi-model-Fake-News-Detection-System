from pathlib import Path

import pandas as pd
from PIL import Image

from fnews.data.loaders import UnifiedMultimodalDataset
from fnews.data.research_dataset import (
    build_fakeddit,
    build_newsclippings_metadata,
    select_fakeddit_rows,
    validate_resize_encode,
)


def test_resize_encode_validates_and_bounds_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (1000, 500), "blue").save(source)
    result = validate_resize_encode(source, tmp_path / "out.jpg", max_dimension=512, image_format="jpeg", quality=85)
    assert result.width == 512
    assert result.height == 256
    assert Path(result.path).exists()
    assert len(result.sha256) == 64


def test_selection_is_stratified_and_preserves_split() -> None:
    frame = pd.DataFrame(
        {
            "id": list("abcdef"),
            "split": ["train", "train", "train", "train", "val", "test"],
            "2_way_label": [0, 0, 1, 1, 0, 1],
            "valid_image": [True] * 6,
            "local_path": ["x"] * 6,
        }
    )
    selected = select_fakeddit_rows(frame, per_split_cap=2, seed=7)
    assert set(selected["split"]) == {"train", "validate", "test"}
    assert set(selected.loc[selected["split"].eq("train"), "2_way_label"]) == {0, 1}


def test_future_fakeddit_build_translates_source_label_direction(tmp_path: Path) -> None:
    image = tmp_path / "source.jpg"
    Image.new("RGB", (12, 8), "green").save(image)
    source = pd.DataFrame(
        {
            "id": ["source-true", "source-non-true"],
            "split": ["train", "train"],
            "2_way_label": [1, 0],
            "clean_title": ["Verified source true", "Source non true"],
            "image_url": ["https://example.test/one.jpg", "https://example.test/two.jpg"],
            "local_path": [str(image), str(image)],
        }
    )
    output = tmp_path / "out"
    output.mkdir()
    built, failures = build_fakeddit(
        source, output, cap_bytes=1_000_000, min_free_bytes=0, max_dimension=32, image_format="jpeg", quality=85
    )
    assert not failures
    assert built.set_index("original_id").loc["source-true", "fake_news_label"] == 0
    assert built.set_index("original_id").loc["source-non-true", "fake_news_label"] == 1


def test_newsclippings_namespace_is_not_fake_news() -> None:
    source = pd.DataFrame({"sample_id": ["a"], "caption_id": [1], "image_id": [2], "falsified": [1], "original_split": ["val"]})
    result = build_newsclippings_metadata(source)
    assert pd.isna(result.loc[0, "fake_news_label"])
    assert result.loc[0, "image_caption_mismatch_label"] == 1
    assert result.loc[0, "download_status"] == "blocked_missing_visualnews_metadata_and_image"


def test_loader_excludes_unverified_rows(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "one.jpg"
    image_path.parent.mkdir()
    Image.new("RGB", (4, 4), "green").save(image_path)
    manifest = pd.DataFrame(
        {
            "task_type": ["fake_news_classification", "image_caption_mismatch"],
            "download_status": ["verified_reused_or_processed", "blocked_missing_visualnews_metadata_and_image"],
            "image_path": ["images/one.jpg", pd.NA],
            "text": ["text", pd.NA],
            "fake_news_label": [1, pd.NA],
            "image_caption_mismatch_label": [pd.NA, 0],
            "sample_id": ["one", "two"],
        }
    )
    path = tmp_path / "manifests" / "train.parquet"
    path.parent.mkdir()
    manifest.to_parquet(path, index=False)
    dataset = UnifiedMultimodalDataset(path, "fake_news_classification")
    assert len(dataset) == 1
    assert dataset[0]["label"] == 1
