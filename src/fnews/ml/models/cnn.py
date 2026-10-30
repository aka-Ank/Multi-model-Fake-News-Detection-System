from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fnews.ml.metrics.classification import binary_classification_metrics
from fnews.ml.nn.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Layer,
    MaxPool2D,
    ReLU,
    Sigmoid,
)
from fnews.ml.nn.losses import binary_cross_entropy, binary_cross_entropy_backward
from fnews.ml.nn.optimizers import SGD, Adam, Optimizer, SGDMomentum


@dataclass
class ManualCNNClassifier:
    image_channels: int = 3
    learning_rate: float = 0.001
    dropout: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        self.layers: list[Layer] = [
            Conv2D(self.image_channels, 8, 3, padding=1, seed=self.seed),
            ReLU(),
            MaxPool2D(2, 2),
            Conv2D(8, 16, 3, padding=1, seed=self.seed + 1),
            ReLU(),
            MaxPool2D(2, 2),
            Conv2D(16, 32, 3, padding=1, seed=self.seed + 2),
            ReLU(),
            GlobalAveragePooling2D(),
            Dense(32, 32, seed=self.seed + 3, initialization="he"),
            ReLU(),
            Dropout(self.dropout, seed=self.seed + 4),
            Dense(32, 1, seed=self.seed + 5, initialization="xavier"),
            Sigmoid(),
        ]
        self.optimizer = Adam(learning_rate=self.learning_rate)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        output = self.encode_image(inputs, training=training)
        for layer in self.layers[9:]:
            output = layer.forward(output, training=training)
        return output

    def encode_image(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        output = np.asarray(inputs, dtype=np.float64)
        for layer in self.layers[:9]:
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

    def predict_proba(self, inputs: np.ndarray, batch_size: int = 64) -> np.ndarray:
        scores = []
        for start in range(0, len(inputs), batch_size):
            scores.append(self.forward(inputs[start : start + batch_size], training=False).reshape(-1))
        return np.concatenate(scores)

    def predict(self, inputs: np.ndarray, threshold: float = 0.5, batch_size: int = 64) -> np.ndarray:
        return (self.predict_proba(inputs, batch_size=batch_size) >= threshold).astype(int)

    def fit(
        self,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        *,
        validation_images: np.ndarray | None = None,
        validation_labels: np.ndarray | None = None,
        epochs: int = 10,
        batch_size: int = 16,
    ) -> list[dict[str, float]]:
        rng = np.random.default_rng(self.seed)
        train_labels = np.asarray(train_labels, dtype=np.float64).reshape(-1, 1)
        history: list[dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            indices = np.arange(len(train_images))
            rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                probs = self.forward(train_images[batch_indices], training=True)
                grad = binary_cross_entropy_backward(probs, train_labels[batch_indices])
                self.backward(grad)
                self.optimizer.step(self.params_and_grads())

            train_scores = self.predict_proba(train_images, batch_size=batch_size)
            train_pred = (train_scores >= 0.5).astype(int)
            row = {
                "epoch": float(epoch),
                "train_loss": binary_cross_entropy(train_scores, train_labels.reshape(-1)),
                "train_accuracy": binary_classification_metrics(train_labels.reshape(-1), train_pred)["accuracy"],
                "train_macro_f1": binary_classification_metrics(train_labels.reshape(-1), train_pred)["macro_f1"],
            }
            if validation_images is not None and validation_labels is not None:
                validation_scores = self.predict_proba(validation_images, batch_size=batch_size)
                validation_pred = (validation_scores >= 0.5).astype(int)
                validation_metrics = binary_classification_metrics(
                    validation_labels,
                    validation_pred,
                    validation_scores,
                )
                row.update(
                    {
                        "validation_loss": binary_cross_entropy(validation_scores, validation_labels),
                        "validation_accuracy": validation_metrics["accuracy"],
                        "validation_precision": validation_metrics["precision"],
                        "validation_recall": validation_metrics["recall"],
                        "validation_macro_f1": validation_metrics["macro_f1"],
                        "validation_weighted_f1": validation_metrics["weighted_f1"],
                        "validation_roc_auc": validation_metrics["roc_auc"],
                    }
                )
            history.append(row)
        return history

    def parameter_summary(self) -> list[dict[str, object]]:
        rows = []
        for index, layer in enumerate(self.layers):
            params = layer.parameters()
            count = int(sum(param.size for param in params))
            if count:
                rows.append({"index": index, "layer": layer.__class__.__name__, "parameters": count})
        rows.append({"index": "total", "layer": "total", "parameters": sum(int(row["parameters"]) for row in rows)})
        return rows

    def save(self, directory: str | Path, metadata: dict[str, object] | None = None) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for index, layer in enumerate(self.layers):
            if isinstance(layer, (Conv2D, Dense)):
                arrays[f"layer_{index}_weights"] = layer.weights
                arrays[f"layer_{index}_bias"] = layer.bias
        np.savez(path / "model.npz", **arrays)
        payload = {
            "architecture": "conv8_pool_conv16_pool_conv32_gap_dense32",
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "seed": self.seed,
            "parameter_summary": self.parameter_summary(),
            "metadata": metadata or {},
        }
        (path / "model_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def get_state(self) -> dict[str, np.ndarray]:
        state = {}
        for index, layer in enumerate(self.layers):
            if isinstance(layer, (Conv2D, Dense)):
                state[f"layer_{index}_weights"] = layer.weights.copy()
                state[f"layer_{index}_bias"] = layer.bias.copy()
        return state

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        for index, layer in enumerate(self.layers):
            if isinstance(layer, (Conv2D, Dense)):
                layer.weights = state[f"layer_{index}_weights"].copy()
                layer.bias = state[f"layer_{index}_bias"].copy()

    @classmethod
    def load(cls, directory: str | Path) -> ManualCNNClassifier:
        path = Path(directory)
        payload = json.loads((path / "model_metadata.json").read_text(encoding="utf-8"))
        arrays = np.load(path / "model.npz")
        model = cls(
            learning_rate=float(payload["learning_rate"]),
            dropout=float(payload["dropout"]),
            seed=int(payload["seed"]),
        )
        for index, layer in enumerate(model.layers):
            if isinstance(layer, (Conv2D, Dense)):
                layer.weights = arrays[f"layer_{index}_weights"]
                layer.bias = arrays[f"layer_{index}_bias"]
        return model


@dataclass
class ManualFlattenCNNClassifier:
    image_channels: int = 3
    image_size: int = 64
    learning_rate: float = 0.001
    seed: int = 42
    optimizer_name: str = "adam"

    def __post_init__(self) -> None:
        pooled_size = self.image_size // 4
        self.layers: list[Layer] = [
            Conv2D(self.image_channels, 4, 3, padding=1, seed=self.seed),
            ReLU(),
            MaxPool2D(2, 2),
            Conv2D(4, 8, 3, padding=1, seed=self.seed + 1),
            ReLU(),
            MaxPool2D(2, 2),
            Flatten(),
            Dense(8 * pooled_size * pooled_size, 32, seed=self.seed + 2, initialization="he"),
            ReLU(),
            Dense(32, 1, seed=self.seed + 3, initialization="xavier"),
            Sigmoid(),
        ]
        self.optimizer = build_optimizer(self.optimizer_name, self.learning_rate)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        output = self.encode_image(inputs, training=training)
        for layer in self.layers[9:]:
            output = layer.forward(output, training=training)
        return output

    def encode_image(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        output = np.asarray(inputs, dtype=np.float64)
        for layer in self.layers[:9]:
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

    def predict_proba(self, inputs: np.ndarray, batch_size: int = 64) -> np.ndarray:
        scores = []
        for start in range(0, len(inputs), batch_size):
            scores.append(self.forward(inputs[start : start + batch_size], training=False).reshape(-1))
        return np.concatenate(scores)

    def fit(
        self,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        *,
        epochs: int = 100,
        batch_size: int | None = None,
        shuffle: bool = False,
        stop_accuracy: float | None = 0.95,
    ) -> list[dict[str, float]]:
        if batch_size is None:
            batch_size = len(train_images)
        rng = np.random.default_rng(self.seed)
        labels = np.asarray(train_labels, dtype=np.float64).reshape(-1, 1)
        history: list[dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            indices = np.arange(len(train_images))
            if shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                probs = self.forward(train_images[batch_indices], training=True)
                grad = binary_cross_entropy_backward(probs, labels[batch_indices])
                self.backward(grad)
                self.optimizer.step(self.params_and_grads())
            scores = self.predict_proba(train_images, batch_size=batch_size)
            predictions = (scores >= 0.5).astype(int)
            metrics = binary_classification_metrics(train_labels, predictions, scores)
            row = {
                "epoch": float(epoch),
                "train_loss": binary_cross_entropy(scores, train_labels),
                "train_accuracy": metrics["accuracy"],
                "train_macro_f1": metrics["macro_f1"],
                "train_roc_auc": metrics["roc_auc"],
            }
            history.append(row)
            if stop_accuracy is not None and row["train_accuracy"] >= stop_accuracy:
                break
        return history

    def parameter_summary(self) -> list[dict[str, object]]:
        rows = []
        for index, layer in enumerate(self.layers):
            params = layer.parameters()
            count = int(sum(param.size for param in params))
            if count:
                rows.append({"index": index, "layer": layer.__class__.__name__, "parameters": count})
        rows.append({"index": "total", "layer": "total", "parameters": sum(int(row["parameters"]) for row in rows)})
        return rows

    def save(self, directory: str | Path, metadata: dict[str, object] | None = None) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for index, layer in enumerate(self.layers):
            if isinstance(layer, (Conv2D, Dense)):
                arrays[f"layer_{index}_weights"] = layer.weights
                arrays[f"layer_{index}_bias"] = layer.bias
        np.savez(path / "model.npz", **arrays)
        payload = {
            "architecture": "conv4_pool_conv8_pool_flatten_dense32",
            "image_size": self.image_size,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "optimizer_name": self.optimizer_name,
            "parameter_summary": self.parameter_summary(),
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
