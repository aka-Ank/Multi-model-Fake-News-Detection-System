from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fnews.ml.metrics.classification import binary_classification_metrics
from fnews.ml.nn.layers import Dense, Dropout, Layer, ReLU, Sigmoid
from fnews.ml.nn.losses import binary_cross_entropy, binary_cross_entropy_backward
from fnews.ml.nn.optimizers import SGD, Adam, Optimizer, SGDMomentum


@dataclass
class ManualMLPClassifier:
    input_dim: int
    hidden_dims: tuple[int, ...] = (32,)
    learning_rate: float = 0.001
    dropout: float = 0.0
    seed: int = 42
    optimizer_name: str = "adam"

    def __post_init__(self) -> None:
        self.layers: list[Layer] = []
        previous = self.input_dim
        for layer_index, hidden_dim in enumerate(self.hidden_dims):
            self.layers.append(
                Dense(previous, hidden_dim, seed=self.seed + layer_index, initialization="he")
            )
            self.layers.append(ReLU())
            if self.dropout > 0:
                self.layers.append(Dropout(self.dropout, seed=self.seed + 100 + layer_index))
            previous = hidden_dim
        self.layers.append(Dense(previous, 1, seed=self.seed + 999, initialization="xavier"))
        self.layers.append(Sigmoid())
        self.optimizer = build_optimizer(self.optimizer_name, self.learning_rate)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        output = np.asarray(inputs, dtype=np.float64)
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output

    def backward(self, grad_output: np.ndarray) -> None:
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for layer in self.layers:
            pairs.extend(zip(layer.parameters(), layer.gradients(), strict=True))
        return pairs

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs, training=False).reshape(-1)

    def predict(self, inputs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(inputs) >= threshold).astype(int)

    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        *,
        validation_features: np.ndarray | None = None,
        validation_labels: np.ndarray | None = None,
        epochs: int = 20,
        batch_size: int = 64,
    ) -> list[dict[str, float]]:
        rng = np.random.default_rng(self.seed)
        train_labels = np.asarray(train_labels, dtype=np.float64).reshape(-1, 1)
        history: list[dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            indices = np.arange(len(train_features))
            rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                probs = self.forward(train_features[batch_indices], training=True)
                grad = binary_cross_entropy_backward(probs, train_labels[batch_indices])
                self.backward(grad)
                self.optimizer.step(self.params_and_grads())

            train_probs = self.predict_proba(train_features)
            train_pred = (train_probs >= 0.5).astype(int)
            row = {
                "epoch": float(epoch),
                "train_loss": binary_cross_entropy(train_probs, train_labels.reshape(-1)),
                "train_macro_f1": binary_classification_metrics(train_labels.reshape(-1), train_pred)[
                    "macro_f1"
                ],
            }
            if validation_features is not None and validation_labels is not None:
                validation_probs = self.predict_proba(validation_features)
                validation_pred = (validation_probs >= 0.5).astype(int)
                row["validation_loss"] = binary_cross_entropy(validation_probs, validation_labels)
                row["validation_macro_f1"] = binary_classification_metrics(
                    validation_labels,
                    validation_pred,
                )["macro_f1"]
            history.append(row)
        return history

    def save(self, directory: str | Path, metadata: dict[str, object] | None = None) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {}
        dense_index = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                arrays[f"dense_{dense_index}_weights"] = layer.weights
                arrays[f"dense_{dense_index}_bias"] = layer.bias
                dense_index += 1
        np.savez(path / "model.npz", **arrays)
        payload = {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "seed": self.seed,
            "optimizer_name": self.optimizer_name,
            "metadata": metadata or {},
        }
        (path / "model_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_optimizer(name: str, learning_rate: float) -> Optimizer:
    if name == "sgd":
        return SGD(learning_rate=learning_rate)
    if name == "momentum":
        return SGDMomentum(learning_rate=learning_rate)
    if name == "adam":
        return Adam(learning_rate=learning_rate)
    raise ValueError(f"Unknown optimizer: {name}")
