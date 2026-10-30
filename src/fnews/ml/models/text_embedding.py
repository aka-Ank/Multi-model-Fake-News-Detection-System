from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fnews.ml.metrics.classification import binary_classification_metrics
from fnews.ml.nn.layers import Dense, Dropout, Embedding, MaskedMeanPooling, ReLU, Sigmoid
from fnews.ml.nn.losses import binary_cross_entropy, binary_cross_entropy_backward
from fnews.ml.nn.optimizers import Adam


@dataclass
class ManualTextEmbeddingClassifier:
    vocab_size: int
    embedding_dim: int = 64
    hidden_dim: int = 64
    pad_index: int = 0
    learning_rate: float = 0.001
    dropout: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        self.embedding = Embedding(self.vocab_size, self.embedding_dim, pad_index=self.pad_index, seed=self.seed)
        self.pooling = MaskedMeanPooling(pad_index=self.pad_index)
        self.layers = [
            Dense(self.embedding_dim, self.hidden_dim, seed=self.seed + 1, initialization="he"),
            ReLU(),
            Dropout(self.dropout, seed=self.seed + 2),
            Dense(self.hidden_dim, 1, seed=self.seed + 3),
            Sigmoid(),
        ]
        self.optimizer = Adam(learning_rate=self.learning_rate)

    def forward(self, token_ids: np.ndarray, training: bool = True) -> np.ndarray:
        output = self.encode_text(token_ids, training=training)
        for layer in self.layers[3:]:
            output = layer.forward(output, training=training)
        return output

    def encode_text(self, token_ids: np.ndarray, training: bool = False) -> np.ndarray:
        embedded = self.embedding.forward(token_ids, training=training)
        output = self.pooling.forward(embedded, training=training, token_ids=token_ids)
        for layer in self.layers[:2]:
            output = layer.forward(output, training=training)
        return output

    def backward(self, grad_output: np.ndarray) -> None:
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        grad = self.pooling.backward(grad)
        self.embedding.backward(grad)

    def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        pairs = list(zip(self.embedding.parameters(), self.embedding.gradients(), strict=True))
        for layer in self.layers:
            pairs.extend(zip(layer.parameters(), layer.gradients(), strict=True))
        return pairs

    def predict_proba(self, token_ids: np.ndarray, batch_size: int = 512) -> np.ndarray:
        scores = []
        for start in range(0, len(token_ids), batch_size):
            scores.append(self.forward(token_ids[start : start + batch_size], training=False).reshape(-1))
        return np.concatenate(scores)

    def fit(
        self,
        train_tokens: np.ndarray,
        train_labels: np.ndarray,
        *,
        validation_tokens: np.ndarray | None = None,
        validation_labels: np.ndarray | None = None,
        epochs: int = 5,
        batch_size: int = 128,
        shuffle: bool = True,
        stop_accuracy: float | None = None,
        early_stopping_patience: int | None = None,
        restore_best: bool = False,
    ) -> list[dict[str, float]]:
        rng = np.random.default_rng(self.seed)
        labels = np.asarray(train_labels, dtype=np.float64).reshape(-1, 1)
        history: list[dict[str, float]] = []
        best_validation_f1 = -np.inf
        best_parameters: list[np.ndarray] | None = None
        patience_used = 0
        self.best_epoch_: int | None = None
        for epoch in range(1, epochs + 1):
            indices = np.arange(len(train_tokens))
            if shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                probs = self.forward(train_tokens[batch_indices], training=True)
                grad = binary_cross_entropy_backward(probs, labels[batch_indices])
                self.backward(grad)
                self.optimizer.step(self.params_and_grads())
                self.embedding.weights[self.pad_index] = 0.0

            train_scores = self.predict_proba(train_tokens, batch_size=batch_size)
            train_pred = (train_scores >= 0.5).astype(int)
            train_metrics = binary_classification_metrics(train_labels, train_pred, train_scores)
            row = {
                "epoch": float(epoch),
                "train_loss": binary_cross_entropy(train_scores, train_labels),
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
            }
            if validation_tokens is not None and validation_labels is not None:
                val_scores = self.predict_proba(validation_tokens, batch_size=batch_size)
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
                if row["validation_macro_f1"] > best_validation_f1:
                    best_validation_f1 = row["validation_macro_f1"]
                    best_parameters = [parameter.copy() for parameter, _ in self.params_and_grads()]
                    self.best_epoch_ = epoch
                    patience_used = 0
                else:
                    patience_used += 1
            history.append(row)
            if stop_accuracy is not None and row["train_accuracy"] >= stop_accuracy:
                break
            if (
                early_stopping_patience is not None
                and validation_tokens is not None
                and validation_labels is not None
                and patience_used >= early_stopping_patience
            ):
                break
        if restore_best and best_parameters is not None:
            for (parameter, _), best_parameter in zip(
                self.params_and_grads(), best_parameters, strict=True
            ):
                parameter[...] = best_parameter
        return history

    def parameter_summary(self) -> list[dict[str, object]]:
        rows = [{"layer": "Embedding", "parameters": int(self.embedding.weights.size)}]
        for layer in self.layers:
            count = int(sum(param.size for param in layer.parameters()))
            if count:
                rows.append({"layer": layer.__class__.__name__, "parameters": count})
        rows.append({"layer": "total", "parameters": sum(int(row["parameters"]) for row in rows)})
        return rows

    def save(self, directory: str | Path, metadata: dict[str, object] | None = None) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {"embedding": self.embedding.weights}
        dense_index = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                arrays[f"dense_{dense_index}_weights"] = layer.weights
                arrays[f"dense_{dense_index}_bias"] = layer.bias
                dense_index += 1
        np.savez(path / "model.npz", **arrays)
        payload = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "pad_index": self.pad_index,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "seed": self.seed,
            "parameter_summary": self.parameter_summary(),
            "metadata": metadata or {},
        }
        (path / "model_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> ManualTextEmbeddingClassifier:
        path = Path(directory)
        payload = json.loads((path / "model_metadata.json").read_text(encoding="utf-8"))
        arrays = np.load(path / "model.npz")
        model = cls(
            vocab_size=int(payload["vocab_size"]),
            embedding_dim=int(payload["embedding_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            pad_index=int(payload["pad_index"]),
            learning_rate=float(payload["learning_rate"]),
            dropout=float(payload["dropout"]),
            seed=int(payload["seed"]),
        )
        model.embedding.weights = arrays["embedding"]
        dense_index = 0
        for layer in model.layers:
            if isinstance(layer, Dense):
                layer.weights = arrays[f"dense_{dense_index}_weights"]
                layer.bias = arrays[f"dense_{dense_index}_bias"]
                dense_index += 1
        return model
