from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fnews.ml.metrics.classification import binary_classification_metrics
from fnews.ml.nn.activations import sigmoid
from fnews.ml.nn.layers import Dense, Dropout, ReLU, Sigmoid
from fnews.ml.nn.losses import binary_cross_entropy, binary_cross_entropy_backward
from fnews.ml.nn.optimizers import Adam


@dataclass
class FusionHead:
    input_dim: int
    hidden_dim: int = 64
    learning_rate: float = 0.001
    dropout: float = 0.2
    seed: int = 42

    def __post_init__(self) -> None:
        self.layers = [
            Dense(self.input_dim, self.hidden_dim, seed=self.seed, initialization="he"),
            ReLU(),
            Dropout(self.dropout, seed=self.seed + 1),
            Dense(self.hidden_dim, 1, seed=self.seed + 2),
            Sigmoid(),
        ]
        self.optimizer = Adam(learning_rate=self.learning_rate)

    def forward(self, features: np.ndarray, training: bool = True) -> np.ndarray:
        output = features
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output

    def backward(self, grad_output: np.ndarray) -> None:
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        pairs = []
        for layer in self.layers:
            pairs.extend(zip(layer.parameters(), layer.gradients(), strict=True))
        return pairs

    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        *,
        validation_features: np.ndarray | None = None,
        validation_labels: np.ndarray | None = None,
        epochs: int = 30,
        batch_size: int = 128,
        shuffle: bool = True,
    ) -> list[dict[str, float]]:
        rng = np.random.default_rng(self.seed)
        labels = train_labels.reshape(-1, 1).astype(float)
        history = []
        for epoch in range(1, epochs + 1):
            indices = np.arange(len(train_features))
            if shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                probs = self.forward(train_features[batch], training=True)
                grad = binary_cross_entropy_backward(probs, labels[batch])
                self.backward(grad)
                self.optimizer.step(self.params_and_grads())
            train_scores = self.predict_proba(train_features, batch_size=batch_size)
            train_pred = (train_scores >= 0.5).astype(int)
            train_metrics = binary_classification_metrics(train_labels, train_pred, train_scores)
            row = {
                "epoch": float(epoch),
                "train_loss": binary_cross_entropy(train_scores, train_labels),
                "train_macro_f1": train_metrics["macro_f1"],
            }
            if validation_features is not None and validation_labels is not None:
                val_scores = self.predict_proba(validation_features, batch_size=batch_size)
                val_pred = (val_scores >= 0.5).astype(int)
                val_metrics = binary_classification_metrics(validation_labels, val_pred, val_scores)
                row.update(
                    {
                        "validation_loss": binary_cross_entropy(val_scores, validation_labels),
                        "validation_accuracy": val_metrics["accuracy"],
                        "validation_precision": val_metrics["precision"],
                        "validation_recall": val_metrics["recall"],
                        "validation_macro_f1": val_metrics["macro_f1"],
                        "validation_weighted_f1": val_metrics["weighted_f1"],
                        "validation_roc_auc": val_metrics["roc_auc"],
                    }
                )
            history.append(row)
        return history

    def predict_proba(self, features: np.ndarray, batch_size: int = 512) -> np.ndarray:
        scores = []
        for start in range(0, len(features), batch_size):
            scores.append(self.forward(features[start : start + batch_size], training=False).reshape(-1))
        return np.concatenate(scores)


def concatenate_features(text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
    return np.concatenate([text_features, image_features], axis=1)


def similarity_features(text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
    dim = min(text_features.shape[1], image_features.shape[1])
    text_slice = text_features[:, :dim]
    image_slice = image_features[:, :dim]
    text_norm = text_slice / (np.linalg.norm(text_slice, axis=1, keepdims=True) + 1e-12)
    image_norm = image_slice / (np.linalg.norm(image_slice, axis=1, keepdims=True) + 1e-12)
    cosine = (text_norm * image_norm).sum(axis=1, keepdims=True)
    return np.concatenate([text_features, image_features, cosine], axis=1)


def gated_features(text_features: np.ndarray, image_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dim = min(text_features.shape[1], image_features.shape[1])
    text_slice = text_features[:, :dim]
    image_slice = image_features[:, :dim]
    gate = sigmoid(text_slice - image_slice)
    fused = gate * text_slice + (1 - gate) * image_slice
    return fused, gate


def validate_alignment(ids_a: np.ndarray, ids_b: np.ndarray, labels_a: np.ndarray, labels_b: np.ndarray) -> None:
    if not np.array_equal(ids_a.astype(str), ids_b.astype(str)):
        raise ValueError("Feature ID alignment failed")
    if not np.array_equal(labels_a.astype(int), labels_b.astype(int)):
        raise ValueError("Feature label alignment failed")
