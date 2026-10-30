from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REQUIRED_ANNOTATION_FIELDS = {"id", "image_id", "falsified"}
MANIFEST_COLUMNS = [
    "sample_id",
    "dataset_source",
    "original_split",
    "caption_id",
    "image_id",
    "caption",
    "image_path",
    "falsified",
    "source_dataset",
    "similarity_score",
    "image_checksum",
    "validation_status",
    "download_status",
]


def load_annotation(path: Path, original_split: str, source_dataset: str) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations", [])
    frame = pd.DataFrame(annotations)
    missing = REQUIRED_ANNOTATION_FIELDS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing annotation fields: {sorted(missing)}")
    result = pd.DataFrame(
        {
            "sample_id": [f"{source_dataset}:{original_split}:{index}" for index in range(len(frame))],
            "dataset_source": "NewsCLIPpings",
            "original_split": original_split,
            "caption_id": frame["id"],
            "image_id": frame["image_id"],
            "caption": None,
            "image_path": None,
            "falsified": frame["falsified"].astype("int8"),
            "source_dataset": source_dataset,
            "similarity_score": frame.get("similarity_score"),
            "image_checksum": None,
            "validation_status": "unresolved_requires_visualnews",
            "download_status": "annotations_verified_images_missing",
        }
    )
    return result[MANIFEST_COLUMNS]


def build_provisional_manifest(root: Path, output_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("data/*/*.json")):
        frames.append(load_annotation(path, path.stem, path.parent.name))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=MANIFEST_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return result


def split_overlap(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    splits = sorted(frame["original_split"].unique())
    for index, left in enumerate(splits):
        left_values = set(frame.loc[frame["original_split"].eq(left), column].dropna())
        for right in splits[index + 1 :]:
            right_values = set(frame.loc[frame["original_split"].eq(right), column].dropna())
            rows.append({"column": column, "left_split": left, "right_split": right, "overlap": len(left_values & right_values)})
    return pd.DataFrame(rows)
