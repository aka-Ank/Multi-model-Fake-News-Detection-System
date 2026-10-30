from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fnews.config import FakedditConfig


@dataclass(frozen=True)
class SplitFile:
    path: Path
    split: str
    is_multimodal_only: bool


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def discover_tsv_files(config: FakedditConfig) -> list[SplitFile]:
    files: list[SplitFile] = []
    for path in sorted(config.raw_dir.rglob("*.tsv")):
        lower_name = path.as_posix().lower()
        split = infer_split(lower_name, config.split_name_patterns)
        is_multimodal = any(keyword.lower() in lower_name for keyword in config.multimodal_keywords)
        files.append(SplitFile(path=path, split=split, is_multimodal_only=is_multimodal))
    return files


def infer_split(path_text: str, split_patterns: dict[str, tuple[str, ...]]) -> str:
    for split, patterns in split_patterns.items():
        if any(pattern.lower() in path_text for pattern in patterns):
            return split
    return "unknown"


def read_tsv(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", usecols=columns, low_memory=False)


def drop_unnamed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed")]
    return frame.drop(columns=unnamed)


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == "bool":
        return series
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def build_image_inventory(image_dir: Path) -> pd.DataFrame:
    if not image_dir.exists():
        return pd.DataFrame(columns=["image_id", "image_path", "image_suffix", "image_size_bytes"])

    rows = []
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rows.append(
            {
                "image_id": path.stem,
                "image_path": str(path),
                "image_suffix": path.suffix.lower(),
                "image_size_bytes": path.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


def summarize_frame(
    frame: pd.DataFrame,
    *,
    source_path: Path,
    split: str,
    is_multimodal_only: bool,
    config: FakedditConfig,
    image_ids: set[str],
) -> dict[str, object]:
    text_column = config.text_column
    id_column = config.id_column
    has_image_column = config.has_image_column
    image_url_column = config.image_url_column

    summary: dict[str, object] = {
        "source_file": str(source_path),
        "split": split,
        "is_multimodal_only_file": is_multimodal_only,
        "rows": len(frame),
        "columns": len(frame.columns),
    }

    if id_column in frame.columns:
        ids = frame[id_column].astype("string")
        summary["unique_ids"] = int(ids.nunique(dropna=True))
        summary["duplicate_id_rows"] = int(ids.duplicated(keep=False).sum())
        summary["rows_with_local_image_by_id"] = int(ids.isin(image_ids).sum())

    if text_column in frame.columns:
        text = frame[text_column].astype("string")
        summary["missing_clean_text_rows"] = int(text.isna().sum() + text.fillna("").str.strip().eq("").sum())
        summary["median_clean_text_chars"] = float(text.fillna("").str.len().median())

    if has_image_column in frame.columns:
        summary["has_image_rows"] = int(normalize_bool_series(frame[has_image_column]).sum())

    if image_url_column in frame.columns:
        urls = frame[image_url_column].astype("string").fillna("").str.strip()
        summary["rows_with_image_url"] = int(urls.ne("").sum())

    for label_column in config.label_columns:
        if label_column in frame.columns:
            summary[f"{label_column}_missing_rows"] = int(frame[label_column].isna().sum())
            summary[f"{label_column}_unique_values"] = int(frame[label_column].nunique(dropna=True))

    return summary
