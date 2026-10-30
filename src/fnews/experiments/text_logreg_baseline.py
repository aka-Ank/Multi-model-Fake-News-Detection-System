from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fnews.ml.features.tfidf import ManualTfidfVectorizer
from fnews.ml.metrics.classification import (
    binary_classification_metrics,
    confusion_matrix_binary,
)
from fnews.ml.models.logistic_regression import ManualLogisticRegression
from fnews.ml.preprocessing.split import stratified_sample_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manual TF-IDF + logistic regression baseline.")
    parser.add_argument("--config", default="configs/experiments/text_logreg_10k.json")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_fakeddit_split(path: str | Path) -> pd.DataFrame:
    columns = ["id", "clean_title", "2_way_label"]
    frame = pd.read_csv(path, sep="\t", usecols=columns, low_memory=False)
    frame["clean_title"] = frame["clean_title"].astype("string").fillna("")
    frame["2_way_label"] = frame["2_way_label"].astype(int)
    return frame


def run_experiment(config: dict[str, object]) -> Path:
    start_time = time.perf_counter()
    seed = int(config["seed"])
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    train_frame = load_fakeddit_split(str(config["train_tsv"]))
    validation_frame = load_fakeddit_split(str(config["validation_tsv"]))

    sample_size = int(config["sample_size"])
    sample_indices = stratified_sample_indices(train_frame["2_way_label"].to_numpy(), sample_size, seed)
    sampled_train = train_frame.iloc[sample_indices].reset_index(drop=True)

    vectorizer = ManualTfidfVectorizer(
        min_frequency=int(config["min_frequency"]),
        max_vocabulary_size=int(config["max_vocabulary_size"]),
        tokenization_mode=str(config["tokenization_mode"]),
        l2_normalize=bool(config["l2_normalize"]),
    )
    train_vectors = vectorizer.fit_transform(sampled_train["clean_title"].tolist())
    validation_vectors = vectorizer.transform(validation_frame["clean_title"].tolist())

    train_labels = sampled_train["2_way_label"].to_numpy(dtype=int)
    validation_labels = validation_frame["2_way_label"].to_numpy(dtype=int)

    model = ManualLogisticRegression(
        n_features=vectorizer.n_features,
        learning_rate=float(config["learning_rate"]),
        l2=float(config["l2"]),
        seed=seed,
    )
    history = model.fit(
        train_vectors,
        train_labels,
        validation_vectors=validation_vectors,
        validation_labels=validation_labels,
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        gradient_clip=float(config["gradient_clip"]),
        early_stopping_patience=int(config["early_stopping_patience"]),
    )

    validation_scores = model.predict_proba(validation_vectors)
    validation_predictions = (validation_scores >= float(config["threshold"])).astype(int)
    metrics = binary_classification_metrics(validation_labels, validation_predictions, validation_scores)
    confusion = confusion_matrix_binary(validation_labels, validation_predictions)

    sampled_train.to_parquet(output_dir / "train_subset.parquet", index=False)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame(confusion, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    prediction_frame = validation_frame[["id", "2_way_label"]].copy()
    prediction_frame["score_fake"] = validation_scores
    prediction_frame["prediction"] = validation_predictions
    prediction_frame.to_parquet(output_dir / "predictions.parquet", index=False)

    vectorizer.save(output_dir / "vectorizer")
    model.save(
        output_dir / "model",
        metadata={
            "text_field": "clean_title",
            "label": "2_way_label",
            "label_mapping": {"0": "true", "1": "fake"},
        },
    )

    elapsed = time.perf_counter() - start_time
    train_counts = sampled_train["2_way_label"].value_counts().sort_index().to_dict()
    validation_counts = validation_frame["2_way_label"].value_counts().sort_index().to_dict()
    metadata = {
        "config": config,
        "runtime_seconds": elapsed,
        "vocabulary_size": vectorizer.n_features,
        "train_subset_size": len(sampled_train),
        "validation_size": len(validation_frame),
        "train_label_counts": {str(key): int(value) for key, value in train_counts.items()},
        "validation_label_counts": {str(key): int(value) for key, value in validation_counts.items()},
        "validation_metrics": metrics,
        "model_size_bytes": int((output_dir / "model" / "model.npz").stat().st_size),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    Path(str(config["config_copy_path"])).parent.mkdir(parents=True, exist_ok=True)
    Path(str(config["config_copy_path"])).write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    plot_history(pd.DataFrame(history), output_dir / "plots" / "training_curves.png")
    write_summary(output_dir / "summary.md", metadata, history, confusion)
    return output_dir


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


def write_summary(
    path: Path,
    metadata: dict[str, object],
    history: list[dict[str, float]],
    confusion: np.ndarray,
) -> None:
    metrics = metadata["validation_metrics"]
    assert isinstance(metrics, dict)
    lines = [
        "# Manual Text Baseline: TF-IDF + Logistic Regression",
        "",
        f"- Runtime seconds: {metadata['runtime_seconds']:.2f}",
        f"- Vocabulary size: {metadata['vocabulary_size']:,}",
        f"- Train subset size: {metadata['train_subset_size']:,}",
        f"- Validation size: {metadata['validation_size']:,}",
        f"- Model size bytes: {metadata['model_size_bytes']:,}",
        "",
        "## Validation Metrics",
        "",
        "| Metric | Value |",
        "|:--|--:|",
    ]
    for key in ["accuracy", "precision", "recall", "macro_f1", "weighted_f1", "roc_auc", "brier_score"]:
        lines.append(f"| {key} | {metrics[key]:.6f} |")
    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            "| | pred true(0) | pred fake(1) |",
            "|:--|--:|--:|",
            f"| true true(0) | {confusion[0, 0]} | {confusion[0, 1]} |",
            f"| true fake(1) | {confusion[1, 0]} | {confusion[1, 1]} |",
            "",
            "## Epoch History",
            "",
            pd.DataFrame(history).to_markdown(index=False),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = run_experiment(load_config(args.config))
    print(f"Wrote experiment outputs under {output_dir}")


if __name__ == "__main__":
    main()
