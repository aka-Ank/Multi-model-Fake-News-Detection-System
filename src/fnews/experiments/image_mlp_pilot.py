from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fnews.ml.metrics.classification import binary_classification_metrics, confusion_matrix_binary
from fnews.ml.models.mlp import ManualMLPClassifier
from fnews.ml.preprocessing.scaler import ManualStandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manual MLP on handcrafted image features.")
    parser.add_argument("--config", default="configs/experiments/image_mlp_pilot.json")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_experiment(config: dict[str, object]) -> Path:
    started = time.perf_counter()
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    train_frame = pd.read_parquet(str(config["train_features"]))
    validation_frame = pd.read_parquet(str(config["validation_features"]))
    feature_columns = [column for column in train_frame.columns if column not in {"id", "2_way_label"}]
    train_features = train_frame[feature_columns].to_numpy(dtype=np.float64)
    validation_features = validation_frame[feature_columns].to_numpy(dtype=np.float64)
    train_labels = train_frame["2_way_label"].to_numpy(dtype=int)
    validation_labels = validation_frame["2_way_label"].to_numpy(dtype=int)

    scaler = ManualStandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    validation_scaled = scaler.transform(validation_features)

    model = ManualMLPClassifier(
        input_dim=train_scaled.shape[1],
        hidden_dims=tuple(int(value) for value in config["hidden_dims"]),
        learning_rate=float(config["learning_rate"]),
        dropout=float(config["dropout"]),
        seed=int(config["seed"]),
        optimizer_name=str(config["optimizer"]),
    )
    history = model.fit(
        train_scaled,
        train_labels,
        validation_features=validation_scaled,
        validation_labels=validation_labels,
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
    )
    validation_scores = model.predict_proba(validation_scaled)
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
    scaler.save(output_dir / "scaler")
    model.save(
        output_dir / "model",
        metadata={
            "model_name": "manual_mlp_on_handcrafted_image_features",
            "feature_columns": feature_columns,
            "label_mapping": {"0": "true", "1": "fake"},
        },
    )
    plot_history(pd.DataFrame(history), output_dir / "plots" / "training_curves.png")

    metadata = {
        "config": config,
        "runtime_seconds": time.perf_counter() - started,
        "train_size": len(train_frame),
        "validation_size": len(validation_frame),
        "feature_count": len(feature_columns),
        "validation_metrics": metrics,
        "model_size_bytes": int((output_dir / "model" / "model.npz").stat().st_size),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    Path(str(config["config_copy_path"])).parent.mkdir(parents=True, exist_ok=True)
    Path(str(config["config_copy_path"])).write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
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
        "# Manual MLP Pilot: Handcrafted Image Features",
        "",
        f"- Runtime seconds: {metadata['runtime_seconds']:.2f}",
        f"- Train size: {metadata['train_size']:,}",
        f"- Validation size: {metadata['validation_size']:,}",
        f"- Feature count: {metadata['feature_count']:,}",
        f"- Model size bytes: {metadata['model_size_bytes']:,}",
        "- Official test split: not used",
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
