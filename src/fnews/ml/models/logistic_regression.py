from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fnews.ml.features.bow import SparseVector
from fnews.ml.metrics.classification import binary_classification_metrics


def sigmoid_stable(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1 + exp_values)
    return output


@dataclass
class ManualLogisticRegression:
    n_features: int
    learning_rate: float = 0.5
    l2: float = 0.0
    seed: int = 42
    weights: np.ndarray = field(init=False)
    bias: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.weights = np.zeros(self.n_features, dtype=np.float64)

    def logits(self, vectors: list[SparseVector]) -> np.ndarray:
        output = np.empty(len(vectors), dtype=np.float64)
        for row, vector in enumerate(vectors):
            total = self.bias
            for index, value in vector.items():
                if 0 <= index < self.n_features:
                    total += self.weights[index] * value
            output[row] = total
        return output

    def predict_proba(self, vectors: list[SparseVector]) -> np.ndarray:
        return sigmoid_stable(self.logits(vectors))

    def predict(self, vectors: list[SparseVector], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(vectors) >= threshold).astype(int)

    def binary_cross_entropy(self, vectors: list[SparseVector], labels: np.ndarray) -> float:
        labels = np.asarray(labels, dtype=np.float64)
        logits = self.logits(vectors)
        # Stable BCE: max(z,0) - z*y + log(1 + exp(-abs(z)))
        losses = np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
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
        early_stopping_patience: int | None = 3,
    ) -> list[dict[str, float]]:
        rng = np.random.default_rng(self.seed)
        train_labels = np.asarray(train_labels, dtype=np.float64)
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
                )

            row = {
                "epoch": float(epoch),
                "train_loss": self.binary_cross_entropy(train_vectors, train_labels),
            }
            train_prob = self.predict_proba(train_vectors)
            train_pred = (train_prob >= 0.5).astype(int)
            row["train_macro_f1"] = binary_classification_metrics(train_labels, train_pred)["macro_f1"]
            if validation_vectors is not None and validation_labels is not None:
                validation_labels = np.asarray(validation_labels, dtype=np.float64)
                row["validation_loss"] = self.binary_cross_entropy(validation_vectors, validation_labels)
                validation_prob = self.predict_proba(validation_vectors)
                validation_pred = (validation_prob >= 0.5).astype(int)
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
    ) -> None:
        logits = self.logits(batch_vectors)
        probs = sigmoid_stable(logits)
        errors = probs - batch_labels
        grad_w = np.zeros_like(self.weights)
        for error, vector in zip(errors, batch_vectors, strict=True):
            for index, value in vector.items():
                if 0 <= index < self.n_features:
                    grad_w[index] += error * value
        grad_w /= len(batch_vectors)
        grad_w += self.l2 * self.weights
        grad_b = float(np.mean(errors))

        if gradient_clip is not None:
            grad_norm = math.sqrt(float(np.dot(grad_w, grad_w)) + grad_b * grad_b)
            if grad_norm > gradient_clip:
                scale = gradient_clip / (grad_norm + 1e-12)
                grad_w *= scale
                grad_b *= scale

        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

    def save(self, directory: str | Path, metadata: dict[str, object] | None = None) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "model.npz", weights=self.weights, bias=np.array([self.bias], dtype=np.float64))
        payload = {
            "n_features": self.n_features,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "seed": self.seed,
            "metadata": metadata or {},
        }
        (path / "model_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> ManualLogisticRegression:
        path = Path(directory)
        payload = json.loads((path / "model_metadata.json").read_text(encoding="utf-8"))
        arrays = np.load(path / "model.npz")
        model = cls(
            n_features=int(payload["n_features"]),
            learning_rate=float(payload["learning_rate"]),
            l2=float(payload["l2"]),
            seed=int(payload["seed"]),
        )
        model.weights = arrays["weights"]
        model.bias = float(arrays["bias"][0])
        return model
