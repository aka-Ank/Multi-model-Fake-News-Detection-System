from __future__ import annotations

import os
from pathlib import Path


def configured_data_root() -> Path | None:
    value = os.environ.get("FNEWS_DATA_ROOT")
    return Path(value).expanduser() if value else None


def validate_external_data_root(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"FNEWS_DATA_ROOT is unavailable: {resolved}")
    if not os.access(resolved, os.R_OK | os.W_OK):
        raise PermissionError(f"FNEWS_DATA_ROOT is not readable and writable: {resolved}")
    return resolved


def external_dataset_path(*parts: str) -> Path:
    root = configured_data_root()
    if root is None:
        raise RuntimeError("FNEWS_DATA_ROOT must be set before using external datasets")
    return validate_external_data_root(root).joinpath(*parts)
