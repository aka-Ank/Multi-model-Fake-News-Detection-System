from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FakedditConfig:
    raw_dir: Path
    image_dir: Path
    interim_dir: Path
    reports_dir: Path
    text_column: str
    id_column: str
    image_url_column: str
    has_image_column: str
    label_columns: tuple[str, ...]
    split_name_patterns: dict[str, tuple[str, ...]]
    multimodal_keywords: tuple[str, ...]


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def load_fakeddit_config(path: str | Path) -> FakedditConfig:
    data = load_yaml(path)
    dataset = data.get("dataset", {})
    root = Path(path).resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    split_patterns = {
        name: tuple(patterns)
        for name, patterns in dataset.get("split_name_patterns", {}).items()
    }

    return FakedditConfig(
        raw_dir=resolve(dataset["raw_dir"]),
        image_dir=resolve(dataset["image_dir"]),
        interim_dir=resolve(dataset["interim_dir"]),
        reports_dir=resolve(dataset["reports_dir"]),
        text_column=dataset.get("text_column", "clean_title"),
        id_column=dataset.get("id_column", "id"),
        image_url_column=dataset.get("image_url_column", "image_url"),
        has_image_column=dataset.get("has_image_column", "hasImage"),
        label_columns=tuple(dataset.get("label_columns", [])),
        split_name_patterns=split_patterns,
        multimodal_keywords=tuple(dataset.get("multimodal_keywords", [])),
    )
