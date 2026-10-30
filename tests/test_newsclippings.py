import json

from fnews.data.newsclippings.acquisition import safe_remote_url
from fnews.data.newsclippings.manifest import (
    build_provisional_manifest,
    load_annotation,
    split_overlap,
)


def test_annotation_loader_preserves_mismatch_label(tmp_path) -> None:
    path = tmp_path / "train.json"
    path.write_text(json.dumps({"annotations": [{"id": 4, "image_id": 8, "falsified": 1, "similarity_score": 0.2}]}))

    frame = load_annotation(path, "train", "merged_balanced")

    assert frame.loc[0, "falsified"] == 1
    assert frame.loc[0, "caption"] is None
    assert frame.loc[0, "validation_status"] == "unresolved_requires_visualnews"


def test_provisional_manifest_and_split_overlap(tmp_path) -> None:
    for split, image_id in (("train", 8), ("val", 8)):
        path = tmp_path / "data" / "merged_balanced" / f"{split}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"annotations": [{"id": 4, "image_id": image_id, "falsified": 0}]}))
    manifest = build_provisional_manifest(tmp_path, tmp_path / "output.parquet")

    overlaps = split_overlap(manifest, "image_id")

    assert len(manifest) == 2
    assert overlaps.loc[0, "overlap"] == 1


def test_empty_provisional_manifest_retains_schema(tmp_path) -> None:
    manifest = build_provisional_manifest(tmp_path, tmp_path / "output.parquet")

    assert "original_split" in manifest.columns
    assert manifest.empty


def test_safe_remote_url_rejects_local_and_non_https_hosts() -> None:
    assert safe_remote_url("https://huggingface.co/g-luo/news-clippings/resolve/main/data/a.json")
    assert safe_remote_url("https://us.aws.cdn.hf.co/xet-bridge-us/object.json")
    assert not safe_remote_url("file:///tmp/a.json")
    assert not safe_remote_url("http://huggingface.co/a.json")
