from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fnews.ml.features.bow import SparseVector
from fnews.ml.metrics.classification import binary_classification_metrics


@dataclass
class ManualLinearSVM:
    n_features: int
    learning_rate: float = 0.1
    l2: float = 0.0001
    c: float = 1.0
    class_weights: dict[int, float] | None = None
    seed: int = 42
    weights: np.ndarray = field(init=False)
    bias: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.weights = np.zeros(self.n_features, dtype=np.float64)

    def decision_function(self, vectors: list[SparseVector]) -> np.ndarray:
        scores = np.empty(len(vectors), dtype=np.float64)
        for row, vector in enumerate(vectors):
            total = self.bias
            for index, value in vector.items():
                if 0 <= index < self.n_features:
                    total += self.weights[index] * value
            scores[row] = total
        return scores

    def predict(self, vectors: list[SparseVector]) -> np.ndarray:
        return (self.decision_function(vectors) >= 0).astype(int)

    def hinge_loss(self, vectors: list[SparseVector], labels: np.ndarray) -> float:
        signed_labels = labels_to_signed(labels)
        margins = 1 - signed_labels * self.decision_function(vectors)
        losses = np.maximum(0, margins)
        regularization = 0.5 * self.l2 * float(np.dot(self.weights, self.weights))
        return float(np.mean(losses) + regularization)

    def fit(
        self,
        train_vectors: list[SparseVector],
        train_labels: np.ndarray,
        *,
        validation_vectors: list[SparseVector] | None = None,
        validation_labels: np.ndarray | None = None,
        epochs: int = 8,
        batch_size: int = 256,
        gradient_clip: float | None = 5.0,
        learning_rate_decay: float = 0.0,
        early_stopping_patience: int | None = 3,
    ) -> list[dict[str, float]]:
        rng = np.random.default_rng(self.seed)
        train_labels = np.asarray(train_labels, dtype=int)
        history: list[dict[str, float]] = []
        best_macro_f1 = -math.inf
        best_weights = self.weights.copy()
        best_bias = self.bias
        patience_used = 0

        for epoch in range(1, epochs + 1):
            indices = np.arange(len(train_vectors))
            rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                self._train_batch(
                    [train_vectors[int(index)] for index in batch_indices],
                    train_labels[batch_indices],
                    gradient_clip,
                    self.learning_rate / (1.0 + learning_rate_decay * (epoch - 1)),
                )

            row = {
                "epoch": float(epoch),
                "train_loss": self.hinge_loss(train_vectors, train_labels),
            }
            train_pred = self.predict(train_vectors)
            row["train_macro_f1"] = binary_classification_metrics(train_labels, train_pred)["macro_f1"]
            if validation_vectors is not None and validation_labels is not None:
                validation_labels = np.asarray(validation_labels, dtype=int)
                row["validation_loss"] = self.hinge_loss(validation_vectors, validation_labels)
                validation_pred = self.predict(validation_vectors)
                validation_metrics = binary_classification_metrics(validation_labels, validation_pred)
                row["validation_macro_f1"] = validation_metrics["macro_f1"]
                if row["validation_macro_f1"] > best_macro_f1:
                    best_macro_f1 = row["validation_macro_f1"]
                    best_weights = self.weights.copy()
                    best_bias = self.bias
                    patience_used = 0
                else:
                    patience_used += 1
            history.append(row)
            if early_stopping_patience is not None and validation_vectors is not None and patience_used >= early_stopping_patience:
                break

        if validation_vectors is not None:
            self.weights = best_weights
            self.bias = best_bias
        return history

    def _train_batch(
        self,
        batch_vectors: list[SparseVector],
        batch_labels: np.ndarray,
        gradient_clip: float | None,
        learning_rate: float,
    ) -> None:
        signed_labels = labels_to_signed(batch_labels)
        scores = self.decision_function(batch_vectors)
        active = (1 - signed_labels * scores) > 0
        # Objective: 0.5 * l2 * ||w||² + C * mean(sample_weight * hinge_loss).
        grad_w = self.l2 * self.weights
        grad_b = 0.0
        weights = np.asarray(
            [(self.class_weights or {}).get(int(label), 1.0) for label in batch_labels], dtype=float
        )
        for is_active, signed_label, sample_weight, vector in zip(
            active, signed_labels, weights, batch_vectors, strict=True
        ):
            if not is_active:
                continue
            grad_b += -self.c * float(sample_weight) * float(signed_label) / len(batch_vectors)
            for index, value in vector.items():
                if 0 <= index < self.n_features:
                    grad_w[index] += -self.c * float(sample_weight) * float(signed_label) * value / len(batch_vectors)

        if gradient_clip is not None:
            grad_norm = math.sqrt(float(np.dot(grad_w, grad_w)) + grad_b * grad_b)
            if grad_norm > gradient_clip:
                scale = gradient_clip / (grad_norm + 1e-12)
                grad_w *= scale
                grad_b *= scale

        self.weights -= learning_rate * grad_w
        self.bias -= learning_rate * grad_b

    def save(self, directory: str | Path, metadata: dict[str, object] | None = None) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "model.npz", weights=self.weights, bias=np.array([self.bias], dtype=np.float64))
        payload = {
            "n_features": self.n_features,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "c": self.c,
            "class_weights": self.class_weights,
            "seed": self.seed,
            "metadata": metadata or {},
        }
        (path / "model_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> ManualLinearSVM:
        path = Path(directory)
        payload = json.loads((path / "model_metadata.json").read_text(encoding="utf-8"))
        arrays = np.load(path / "model.npz")
        model = cls(
            n_features=int(payload["n_features"]),
            learning_rate=float(payload["learning_rate"]),
            l2=float(payload["l2"]),
            c=float(payload.get("c", 1.0)),
            class_weights={int(key): float(value) for key, value in payload.get("class_weights", {}).items()} or None,
            seed=int(payload["seed"]),
        )
        model.weights = arrays["weights"]
        model.bias = float(arrays["bias"][0])
        return model


def labels_to_signed(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    return np.where(labels == 1, 1.0, -1.0)
