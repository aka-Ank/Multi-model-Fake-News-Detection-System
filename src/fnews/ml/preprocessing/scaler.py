from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ManualStandardScaler:
    epsilon: float = 1e-12
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> ManualStandardScaler:
        values = np.asarray(values, dtype=np.float64)
        self.mean_ = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < self.epsilon] = 1.0
        self.scale_ = scale
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._require_fit()
        assert self.mean_ is not None
        assert self.scale_ is not None
        return (np.asarray(values, dtype=np.float64) - self.mean_) / self.scale_

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        self.fit(values)
        return self.transform(values)

    def save(self, directory: str | Path) -> None:
        self._require_fit()
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        assert self.mean_ is not None
        assert self.scale_ is not None
        np.savez(path / "scaler_arrays.npz", mean=self.mean_, scale=self.scale_)
        (path / "scaler_metadata.json").write_text(
            json.dumps({"epsilon": self.epsilon}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> ManualStandardScaler:
        path = Path(directory)
        metadata = json.loads((path / "scaler_metadata.json").read_text(encoding="utf-8"))
        arrays = np.load(path / "scaler_arrays.npz")
        scaler = cls(epsilon=float(metadata["epsilon"]))
        scaler.mean_ = arrays["mean"]
        scaler.scale_ = arrays["scale"]
        return scaler

    def _require_fit(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted")
