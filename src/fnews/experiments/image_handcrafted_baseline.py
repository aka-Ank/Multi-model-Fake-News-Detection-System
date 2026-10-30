from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from fnews.ml.features.image_features import (
    dense_rows_to_sparse,
    extract_handcrafted_image_features,
    handcrafted_image_feature_names,
)
from fnews.ml.metrics.classification import binary_classification_metrics, confusion_matrix_binary
from fnews.ml.models.logistic_regression import ManualLogisticRegression
from fnews.ml.preprocessing.scaler import ManualStandardScaler
from fnews.ml.preprocessing.split import stratified_sample_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manual handcrafted-image logistic baseline.")
    parser.add_argument("--config", default="configs/experiments/image_handcrafted_pilot.json")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_experiment(config: dict[str, object]) -> Path:
    started = time.perf_counter()
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    manifest = pd.read_parquet(str(config["image_manifest"]))
    valid = manifest[manifest["valid_image"] & manifest["local_path"].notna()].copy()
    train_frame = valid[valid["split"].astype(str) == "train"].reset_index(drop=True)
    validation_frame = valid[valid["split"].astype(str) == "validate"].reset_index(drop=True)

    requested_sample_size = int(config["sample_size"])
    sample_size = min(requested_sample_size, len(train_frame))
    sample_indices = stratified_sample_indices(train_frame["2_way_label"].to_numpy(), sample_size, int(config["seed"]))
    sampled_train = train_frame.iloc[sample_indices].reset_index(drop=True)

    image_size = tuple(int(value) for value in config["image_size"])
    histogram_bins = int(config["histogram_bins"])
    feature_names = handcrafted_image_feature_names(histogram_bins)
    train_features = extract_feature_matrix(sampled_train, image_size, histogram_bins, "train images")
    validation_features = extract_feature_matrix(validation_frame, image_size, histogram_bins, "validation images")

    scaler = ManualStandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    validation_scaled = scaler.transform(validation_features)
    train_vectors = dense_rows_to_sparse(train_scaled)
    validation_vectors = dense_rows_to_sparse(validation_scaled)

    train_labels = sampled_train["2_way_label"].to_numpy(dtype=int)
    validation_labels = validation_frame["2_way_label"].to_numpy(dtype=int)
    model = ManualLogisticRegression(
        n_features=train_scaled.shape[1],
        learning_rate=float(config["learning_rate"]),
        l2=float(config["l2"]),
        seed=int(config["seed"]),
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

    sampled_train.to_parquet(output_dir / "train_images.parquet", index=False)
    validation_frame.to_parquet(output_dir / "validation_images.parquet", index=False)
    feature_frame(train_features, feature_names, sampled_train).to_parquet(output_dir / "train_features.parquet", index=False)
    feature_frame(validation_features, feature_names, validation_frame).to_parquet(
        output_dir / "validation_features.parquet",
        index=False,
    )
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame(confusion, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    prediction_frame = validation_frame[["id", "2_way_label", "local_path"]].copy()
    prediction_frame["score_fake"] = validation_scores
    prediction_frame["prediction"] = validation_predictions
    prediction_frame.to_parquet(output_dir / "predictions.parquet", index=False)

    scaler.save(output_dir / "scaler")
    model.save(
        output_dir / "model",
        metadata={
            "model_name": "handcrafted_image_logistic_regression",
            "feature_names": feature_names,
            "image_size": image_size,
            "histogram_bins": histogram_bins,
            "label_mapping": {"0": "true", "1": "fake"},
        },
    )
    plot_history(pd.DataFrame(history), output_dir / "plots" / "training_curves.png")

    metadata = {
        "config": config,
        "runtime_seconds": time.perf_counter() - started,
        "requested_train_sample_size": requested_sample_size,
        "actual_train_size": len(sampled_train),
        "validation_size": len(validation_frame),
        "feature_count": len(feature_names),
        "train_label_counts": stringify_counts(sampled_train["2_way_label"]),
        "validation_label_counts": stringify_counts(validation_frame["2_way_label"]),
        "validation_metrics": metrics,
        "model_size_bytes": int((output_dir / "model" / "model.npz").stat().st_size),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    Path(str(config["config_copy_path"])).parent.mkdir(parents=True, exist_ok=True)
    Path(str(config["config_copy_path"])).write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(output_dir / "summary.md", metadata, history, confusion)
    return output_dir


def extract_feature_matrix(
    frame: pd.DataFrame,
    image_size: tuple[int, int],
    histogram_bins: int,
    label: str,
) -> np.ndarray:
    features = []
    for path in tqdm(frame["local_path"].tolist(), desc=f"extract {label}"):
        features.append(
            extract_handcrafted_image_features(
                path,
                image_size=image_size,
                histogram_bins=histogram_bins,
            )
        )
    return np.vstack(features).astype(np.float64)


def feature_frame(features: np.ndarray, names: list[str], source_frame: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(features, columns=names)
    frame.insert(0, "id", source_frame["id"].to_numpy())
    frame.insert(1, "2_way_label", source_frame["2_way_label"].to_numpy())
    return frame


def stringify_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().sort_index().to_dict().items()}


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
        "# Manual Image Baseline: Handcrafted Features + Logistic Regression",
        "",
        f"- Runtime seconds: {metadata['runtime_seconds']:.2f}",
        f"- Requested train sample size: {metadata['requested_train_sample_size']:,}",
        f"- Actual train size: {metadata['actual_train_size']:,}",
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
            "This is a pilot image baseline because only locally downloaded valid images are used.",
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
