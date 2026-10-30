from pathlib import Path

import pandas as pd
from PIL import Image

from fnews.data.image_eda_fakeddit import (
    build_image_manifest,
    calculate_sha256,
    find_local_images,
    inspect_image,
)


def write_config(tmp_path: Path, raw_dir: Path, image_dir: Path) -> Path:
    config = tmp_path / "configs" / "fakeddit.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        f"""
dataset:
  raw_dir: {raw_dir}
  image_dir: {image_dir}
  interim_dir: {tmp_path / "data" / "interim" / "fakeddit"}
  reports_dir: {tmp_path / "reports" / "fakeddit"}
  text_column: clean_title
  id_column: id
  image_url_column: image_url
  has_image_column: hasImage
  label_columns:
    - 2_way_label
    - 3_way_label
    - 6_way_label
  split_name_patterns:
    train:
      - train
    validate:
      - validate
    test:
      - test
  multimodal_keywords:
    - multimodal_only_samples
    - multimodal_
""",
        encoding="utf-8",
    )
    return config


def test_inspect_valid_image_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "abc.jpg"
    Image.new("RGB", (12, 8), color="red").save(path)

    inspection = inspect_image("abc", path)

    assert inspection.downloaded is True
    assert inspection.valid_image is True
    assert inspection.width == 12
    assert inspection.height == 8
    assert inspection.aspect_ratio == 1.5
    assert inspection.sha256 == calculate_sha256(path)
    assert inspection.perceptual_hash is not None


def test_inspect_zero_byte_and_html_failures(tmp_path: Path) -> None:
    zero = tmp_path / "zero.jpg"
    zero.write_bytes(b"")
    html = tmp_path / "html.jpg"
    html.write_text("<html><body>not an image</body></html>", encoding="utf-8")

    zero_result = inspect_image("zero", zero)
    html_result = inspect_image("html", html)

    assert zero_result.valid_image is False
    assert zero_result.error_type == "zero_byte"
    assert html_result.valid_image is False
    assert html_result.error_type == "html_response"


def test_find_local_images_resolves_by_id_with_extensions(tmp_path: Path) -> None:
    Image.new("RGB", (5, 5)).save(tmp_path / "one.png")
    Image.new("RGB", (5, 5)).save(tmp_path / "two.jpg")
    (tmp_path / "note.txt").write_text("ignore", encoding="utf-8")

    local = find_local_images(tmp_path)

    assert set(local) == {"one", "two"}
    assert local["one"].suffix == ".png"


def test_build_manifest_marks_missing_and_valid_images(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw" / "fakeddit"
    split_dir = raw_dir / "multimodal_only_samples"
    image_dir = raw_dir / "images"
    split_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)

    Image.new("RGB", (10, 20)).save(image_dir / "has_image.jpg")
    frame = pd.DataFrame(
        [
            {
                "id": "has_image",
                "clean_title": "valid sample",
                "2_way_label": 0,
                "3_way_label": 0,
                "6_way_label": 0,
                "image_url": "https://example.com/has_image.jpg",
            },
            {
                "id": "missing_image",
                "clean_title": "missing sample",
                "2_way_label": 1,
                "3_way_label": 2,
                "6_way_label": 4,
                "image_url": "https://example.com/missing.jpg",
            },
        ]
    )
    frame.to_csv(split_dir / "multimodal_train.tsv", sep="\t", index=False)
    config = write_config(tmp_path, raw_dir, image_dir)

    manifest = build_image_manifest(str(config), workers=1)

    valid_row = manifest.loc[manifest["id"] == "has_image"].iloc[0]
    missing_row = manifest.loc[manifest["id"] == "missing_image"].iloc[0]
    assert bool(valid_row["downloaded"]) is True
    assert bool(valid_row["valid_image"]) is True
    assert valid_row["width"] == 10
    assert bool(missing_row["downloaded"]) is False
    assert bool(missing_row["valid_image"]) is False
