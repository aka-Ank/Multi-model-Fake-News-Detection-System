from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fnews.experiments.text_logreg_baseline import load_fakeddit_split
from fnews.ml.metrics.classification import binary_classification_metrics, confusion_matrix_binary
from fnews.ml.models.text_embedding import ManualTextEmbeddingClassifier
from fnews.ml.preprocessing.split import stratified_sample_indices
from fnews.ml.preprocessing.text import tokenize

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manual neural text embedding experiment.")
    parser.add_argument("--config", default="configs/experiments/neural_text.json")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_experiment(config: dict[str, object]) -> Path:
    started = time.perf_counter()
    output_dir = Path(str(config["output_dir"]))
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    train_frame_full = load_fakeddit_split(str(config["train_tsv"]))
    validation_frame = load_fakeddit_split(str(config["validation_tsv"]))
    train_indices = stratified_sample_indices(
        train_frame_full["2_way_label"].to_numpy(),
        int(config["sample_size"]),
        int(config["seed"]),
    )
    train_frame = train_frame_full.iloc[train_indices].reset_index(drop=True)
    tokenized_train = [tokenize(text, mode=str(config["tokenization_mode"])) for text in train_frame["clean_title"]]
    tokenized_validation = [
        tokenize(text, mode=str(config["tokenization_mode"])) for text in validation_frame["clean_title"]
    ]
    sequence_stats = sequence_length_stats(tokenized_train)
    max_length = int(config["max_length"]) if config.get("max_length") else int(sequence_stats["p95"])
    vocabulary = build_text_vocabulary(
        tokenized_train,
        min_frequency=int(config["min_frequency"]),
        max_vocabulary_size=int(config["max_vocabulary_size"]),
    )
    train_tokens = encode_documents(tokenized_train, vocabulary, max_length)
    validation_tokens = encode_documents(tokenized_validation, vocabulary, max_length)
    train_labels = train_frame["2_way_label"].to_numpy(dtype=int)
    validation_labels = validation_frame["2_way_label"].to_numpy(dtype=int)

    tiny_history = run_tiny_overfit(train_tokens[:32], train_labels[:32], config, len(vocabulary))
    pd.DataFrame(tiny_history).to_csv(output_dir / "tiny_overfit_history.csv", index=False)

    model = ManualTextEmbeddingClassifier(
        vocab_size=len(vocabulary),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        learning_rate=float(config["learning_rate"]),
        dropout=float(config["dropout"]),
        seed=int(config["seed"]),
    )
    history = model.fit(
        train_tokens,
        train_labels,
        validation_tokens=validation_tokens,
        validation_labels=validation_labels,
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        early_stopping_patience=(
            int(config["early_stopping_patience"])
            if config.get("early_stopping_patience") is not None
            else None
        ),
        restore_best=bool(config.get("restore_best", False)),
    )
    validation_scores = model.predict_proba(validation_tokens, batch_size=int(config["batch_size"]))
    validation_predictions = (validation_scores >= float(config["threshold"])).astype(int)
    metrics = binary_classification_metrics(validation_labels, validation_predictions, validation_scores)
    confusion = confusion_matrix_binary(validation_labels, validation_predictions)

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame(confusion, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    prediction_frame = validation_frame[["id", "2_way_label"]].copy()
    prediction_frame["score_fake"] = validation_scores
    prediction_frame["prediction"] = validation_predictions
    prediction_frame.to_parquet(output_dir / "predictions.parquet", index=False)
    save_vocabulary(vocabulary, output_dir / "vocabulary.json")
    model.save(
        output_dir / "model",
        metadata={
            "tokenization_mode": config["tokenization_mode"],
            "max_length": max_length,
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN,
            "class_mapping": {"0": "true", "1": "fake"},
        },
    )
    plot_history(pd.DataFrame(history), figures_dir / "training_curve.png")

    diagnostics = token_diagnostics(train_tokens, validation_tokens, vocabulary)
    embedding_similarities = inspect_embedding_similarities(model, vocabulary)
    pd.DataFrame(embedding_similarities).to_csv(output_dir / "embedding_similarities.csv", index=False)
    comparison = comparison_table(metrics, int(config["sample_size"]))
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    metadata = {
        "config": config,
        "runtime_seconds": time.perf_counter() - started,
        "sequence_length_stats": sequence_stats,
        "max_length": max_length,
        "vocabulary_size": len(vocabulary),
        "embedding_dim": int(config["embedding_dim"]),
        "parameter_summary": model.parameter_summary(),
        "parameter_count": int(model.parameter_summary()[-1]["parameters"]),
        "tiny_overfit_final": tiny_history[-1],
        "best_epoch": model.best_epoch_,
        "token_diagnostics": diagnostics,
        "validation_metrics": metrics,
        "model_size_bytes": int((output_dir / "model" / "model.npz").stat().st_size),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    Path(str(config["config_copy_path"])).parent.mkdir(parents=True, exist_ok=True)
    Path(str(config["config_copy_path"])).write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    write_model_summary(output_dir / "model_summary.txt", model.parameter_summary())
    write_readme(output_dir / "README.md", metadata, comparison)
    return output_dir


def build_text_vocabulary(
    tokenized_docs: list[list[str]],
    *,
    min_frequency: int,
    max_vocabulary_size: int,
) -> dict[str, int]:
    counts = Counter(token for doc in tokenized_docs for token in doc)
    tokens = [token for token, count in counts.items() if count >= min_frequency]
    tokens.sort(key=lambda token: (-counts[token], token))
    tokens = tokens[: max(0, max_vocabulary_size - 2)]
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    vocab.update({token: index + 2 for index, token in enumerate(tokens)})
    return vocab


def encode_documents(tokenized_docs: list[list[str]], vocabulary: dict[str, int], max_length: int) -> np.ndarray:
    rows = np.zeros((len(tokenized_docs), max_length), dtype=np.int64)
    unk = vocabulary[UNK_TOKEN]
    for row, tokens in enumerate(tokenized_docs):
        for col, token in enumerate(tokens[:max_length]):
            rows[row, col] = vocabulary.get(token, unk)
    return rows


def sequence_length_stats(tokenized_docs: list[list[str]]) -> dict[str, float]:
    lengths = np.array([len(doc) for doc in tokenized_docs], dtype=np.float64)
    return {
        "p50": float(np.percentile(lengths, 50)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "mean": float(lengths.mean()),
    }


def run_tiny_overfit(
    tokens: np.ndarray,
    labels: np.ndarray,
    config: dict[str, object],
    vocab_size: int,
) -> list[dict[str, float]]:
    model = ManualTextEmbeddingClassifier(
        vocab_size=vocab_size,
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        learning_rate=float(config["tiny_learning_rate"]),
        dropout=0.0,
        seed=int(config["seed"]),
    )
    return model.fit(tokens, labels, epochs=int(config["tiny_epochs"]), batch_size=len(tokens), shuffle=False, stop_accuracy=1.0)


def token_diagnostics(train_tokens: np.ndarray, validation_tokens: np.ndarray, vocabulary: dict[str, int]) -> dict[str, float]:
    del vocabulary
    train_nonpad = train_tokens != 0
    validation_nonpad = validation_tokens != 0
    return {
        "train_unk_rate": float(((train_tokens == 1) & train_nonpad).sum() / max(train_nonpad.sum(), 1)),
        "validation_unk_rate": float(((validation_tokens == 1) & validation_nonpad).sum() / max(validation_nonpad.sum(), 1)),
        "train_pad_rate": float((train_tokens == 0).mean()),
        "validation_pad_rate": float((validation_tokens == 0).mean()),
        "train_mean_valid_tokens": float(train_nonpad.sum(axis=1).mean()),
        "validation_mean_valid_tokens": float(validation_nonpad.sum(axis=1).mean()),
    }


def inspect_embedding_similarities(
    model: ManualTextEmbeddingClassifier,
    vocabulary: dict[str, int],
) -> list[dict[str, object]]:
    tokens = [token for token in ["trump", "obama", "police", "cat", "news", "onion"] if token in vocabulary]
    rows = []
    weights = model.embedding.weights
    for i, left in enumerate(tokens):
        for right in tokens[i + 1 :]:
            a = weights[vocabulary[left]]
            b = weights[vocabulary[right]]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            rows.append({"token_a": left, "token_b": right, "cosine": float(np.dot(a, b) / denom) if denom else 0.0})
    return rows


def save_vocabulary(vocabulary: dict[str, int], path: Path) -> None:
    path.write_text(json.dumps(vocabulary, indent=2, sort_keys=True), encoding="utf-8")


def plot_history(history: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["train_loss"], label="train loss")
    if "validation_loss" in history:
        plt.plot(history["epoch"], history["validation_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def comparison_table(metrics: dict[str, float], sample_size: int) -> pd.DataFrame:
    rows = []
    text_aggregate = pd.read_csv("reports/experiments/text_linear_50k_multiseed/aggregate_metrics.csv")
    for _, row in text_aggregate.iterrows():
        rows.append(
            {
                "model": f"TF-IDF + {row['model']} 50k mean",
                "accuracy": row["accuracy_mean"],
                "macro_f1": row["macro_f1_mean"],
                "weighted_f1": row["weighted_f1_mean"],
                "roc_auc": row["roc_auc_mean"],
            }
        )
    rows.append(
        {
            "model": f"Trainable embedding mean-pool {sample_size // 1000}k",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "roc_auc": metrics["roc_auc"],
        }
    )
    return pd.DataFrame(rows)


def write_model_summary(path: Path, parameter_summary: list[dict[str, object]]) -> None:
    lines = ["Manual neural text model parameter summary", ""]
    for row in parameter_summary:
        lines.append(f"{row['layer']}: {row['parameters']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(path: Path, metadata: dict[str, object], comparison: pd.DataFrame) -> None:
    metrics = metadata["validation_metrics"]
    lines = [
        "# Neural Text Baseline",
        "",
        "Trainable embeddings are initialized randomly and learned from scratch. The model uses masked mean pooling so PAD tokens do not affect document representations.",
        "",
        f"- Runtime seconds: {metadata['runtime_seconds']:.2f}",
        f"- Vocabulary size: {metadata['vocabulary_size']}",
        f"- Max sequence length: {metadata['max_length']}",
        f"- Embedding dimension: {metadata['embedding_dim']}",
        f"- Parameter count: {metadata['parameter_count']}",
        f"- Model size bytes: {metadata['model_size_bytes']}",
        f"- Best validation epoch: {metadata['best_epoch']}",
        "",
        "## Tiny Overfit",
        "",
        pd.DataFrame([metadata["tiny_overfit_final"]]).to_markdown(index=False),
        "",
        "## Token Diagnostics",
        "",
        pd.DataFrame([metadata["token_diagnostics"]]).to_markdown(index=False),
        "",
        "## Validation Metrics",
        "",
        f"- Accuracy: {metrics['accuracy']:.6f}",
        f"- Precision: {metrics['precision']:.6f}",
        f"- Recall: {metrics['recall']:.6f}",
        f"- Macro F1: {metrics['macro_f1']:.6f}",
        f"- Weighted F1: {metrics['weighted_f1']:.6f}",
        f"- ROC-AUC: {metrics['roc_auc']:.6f}",
        f"- Brier score: {metrics['brier_score']:.6f}",
        "",
        "## Comparison",
        "",
        comparison.to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = run_experiment(load_config(args.config))
    print(f"Wrote neural text experiment under {output_dir}")


if __name__ == "__main__":
    main()
