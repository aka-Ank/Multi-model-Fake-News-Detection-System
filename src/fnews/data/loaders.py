"""PyTorch dataset loaders for verified rows in the unified manifest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image


class UnifiedMultimodalDataset:
    """Returns only image-text pairs whose bytes were validated by the builder."""

    def __init__(self, manifest_path: str | Path, task_type: str, transform=None) -> None:
        manifest_path = Path(manifest_path).resolve()
        self.dataset_root = manifest_path.parent.parent
        self.frame = pd.read_parquet(manifest_path)
        self.frame = self.frame.loc[
            self.frame["task_type"].eq(task_type)
            & self.frame["download_status"].eq("verified_reused_or_processed")
        ].reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        with Image.open(self.dataset_root / str(row.image_path)) as opened:
            image = opened.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label_column = "fake_news_label" if row.task_type == "fake_news_classification" else "image_caption_mismatch_label"
        return {"image": image, "text": row.text, "label": int(row[label_column]), "sample_id": row.sample_id}


def build_dataloader(manifest_path: str | Path, task_type: str, *, batch_size: int = 16, shuffle: bool = False, transform=None):
    """Create a PyTorch ``DataLoader`` only from verified local pairs."""
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install the image-research extra to use a PyTorch DataLoader.") from exc
    return DataLoader(UnifiedMultimodalDataset(manifest_path, task_type, transform), batch_size=batch_size, shuffle=shuffle)
