from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from fnews.ml.metrics.classification import binary_classification_metrics, confusion_matrix_binary
from fnews.ml.models.cnn import ManualCNNClassifier
from fnews.ml.nn.gradient_check import numerical_gradient, relative_error
from fnews.ml.nn.layers import Conv2D, GlobalAveragePooling2D, MaxPool2D
from fnews.ml.preprocessing.split import stratified_sample_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manual CNN image pilot.")
    parser.add_argument("--config", default="configs/experiments/image_cnn_pilot.json")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_experiment(config: dict[str, object]) -> Path:
    started = time.perf_counter()
    output_dir = Path(str(config["output_dir"]))
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    gradient_checks = run_gradient_checks()
    pd.DataFrame(gradient_checks).to_csv(output_dir / "gradient_checks.csv", index=False)
    if max(row["relative_error"] for row in gradient_checks) > float(config["gradient_check_tolerance"]):
        raise RuntimeError("CNN gradient check failed; refusing to train pilot")

    train_frame = pd.read_parquet(str(config["train_images"]))
    validation_frame = pd.read_parquet(str(config["validation_images"]))
    train_frame = train_frame.iloc[
        stratified_sample_indices(
            train_frame["2_way_label"].to_numpy(),
            min(int(config["train_sample_size"]), len(train_frame)),
            int(config["seed"]),
        )
    ].reset_index(drop=True)
    validation_frame = validation_frame.iloc[
        stratified_sample_indices(
            validation_frame["2_way_label"].to_numpy(),
            min(int(config["validation_sample_size"]), len(validation_frame)),
            int(config["seed"]),
        )
    ].reset_index(drop=True)

    image_size = tuple(int(value) for value in config["image_size"])
    train_pixels = load_pixel_tensor(train_frame, image_size, "load cnn train images")
    validation_pixels = load_pixel_tensor(validation_frame, image_size, "load cnn validation images")
    channel_mean, channel_std = channel_statistics(train_pixels)
    train_pixels = normalize_pixels(train_pixels, channel_mean, channel_std)
    validation_pixels = normalize_pixels(validation_pixels, channel_mean, channel_std)
    train_labels = train_frame["2_way_label"].to_numpy(dtype=int)
    validation_labels = validation_frame["2_way_label"].to_numpy(dtype=int)

    tiny_count = min(int(config["tiny_overfit_size"]), len(train_pixels))
    tiny_model = ManualCNNClassifier(
        learning_rate=float(config["tiny_learning_rate"]),
        dropout=0.0,
        seed=int(config["seed"]),
    )
    tiny_history = tiny_model.fit(
        train_pixels[:tiny_count],
        train_labels[:tiny_count],
        validation_images=train_pixels[:tiny_count],
        validation_labels=train_labels[:tiny_count],
        epochs=int(config["tiny_epochs"]),
        batch_size=int(config["batch_size"]),
    )
    pd.DataFrame(tiny_history).to_csv(output_dir / "tiny_overfit_history.csv", index=False)
    plot_history(pd.DataFrame(tiny_history), figures_dir / "tiny_overfit_curve.png")

    model = ManualCNNClassifier(
        learning_rate=float(config["learning_rate"]),
        dropout=float(config["dropout"]),
        seed=int(config["seed"]),
    )
    history, best_state, best_epoch = train_with_best_state(
        model,
        train_pixels,
        train_labels,
        validation_pixels,
        validation_labels,
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
    )
    model.save(output_dir / "last_model", metadata=metadata_payload(config, train_frame, validation_frame, channel_mean, channel_std))
    model.set_state(best_state)
    model.save(output_dir / "best_model", metadata=metadata_payload(config, train_frame, validation_frame, channel_mean, channel_std))

    validation_scores = model.predict_proba(validation_pixels, batch_size=int(config["batch_size"]))
    validation_predictions = (validation_scores >= float(config["threshold"])).astype(int)
    metrics = binary_classification_metrics(validation_labels, validation_predictions, validation_scores)
    confusion = confusion_matrix_binary(validation_labels, validation_predictions)

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame(model.parameter_summary()).to_csv(output_dir / "parameter_summary.csv", index=False)
    pd.DataFrame(confusion, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    predictions = validation_frame[["id", "2_way_label", "local_path"]].copy()
    predictions["score_fake"] = validation_scores
    predictions["prediction"] = validation_predictions
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    plot_history(pd.DataFrame(history), figures_dir / "training_curve.png")
    write_error_examples(predictions, figures_dir)

    comparison = comparison_table(metrics, output_dir)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    metadata = {
        "config": config,
        "runtime_seconds": time.perf_counter() - started,
        "best_epoch": best_epoch,
        "gradient_checks": gradient_checks,
        "tiny_overfit_final": tiny_history[-1],
        "train_size": len(train_frame),
        "validation_size": len(validation_frame),
        "channel_mean": channel_mean.tolist(),
        "channel_std": channel_std.tolist(),
        "validation_metrics": metrics,
        "parameter_summary": model.parameter_summary(),
        "model_size_bytes": int((output_dir / "best_model" / "model.npz").stat().st_size),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    Path(str(config["config_copy_path"])).parent.mkdir(parents=True, exist_ok=True)
    Path(str(config["config_copy_path"])).write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    write_model_summary(output_dir / "model_summary.txt", model.parameter_summary())
    write_readme(output_dir / "README.md", metadata, comparison, confusion)
    return output_dir


def run_gradient_checks() -> list[dict[str, float | str]]:
    rng = np.random.default_rng(7)
    conv = Conv2D(1, 1, 3, padding=1, seed=8)
    inputs = rng.normal(size=(2, 1, 5, 5))
    grad_output = rng.normal(size=(2, 1, 5, 5))

    def conv_loss() -> float:
        return float(np.sum(conv.forward(inputs) * grad_output))

    conv.forward(inputs)
    grad_input = conv.backward(grad_output)
    conv_weight_error = relative_error(conv.grad_weights, numerical_gradient(conv_loss, conv.weights))
    conv_bias_error = relative_error(conv.grad_bias, numerical_gradient(conv_loss, conv.bias))
    conv_input_error = relative_error(grad_input, numerical_gradient(conv_loss, inputs))

    pool = MaxPool2D(2, 2)
    pool_input = rng.normal(size=(1, 1, 4, 4))
    pool_grad = rng.normal(size=(1, 1, 2, 2))

    def pool_loss() -> float:
        return float(np.sum(pool.forward(pool_input) * pool_grad))

    pool.forward(pool_input)
    pool_input_error = relative_error(pool.backward(pool_grad), numerical_gradient(pool_loss, pool_input))

    gap = GlobalAveragePooling2D()
    gap_input = rng.normal(size=(2, 3, 4, 4))
    gap_grad = rng.normal(size=(2, 3))

    def gap_loss() -> float:
        return float(np.sum(gap.forward(gap_input) * gap_grad))

    gap.forward(gap_input)
    gap_input_error = relative_error(gap.backward(gap_grad), numerical_gradient(gap_loss, gap_input))
    return [
        {"component": "conv2d_weights", "relative_error": conv_weight_error},
        {"component": "conv2d_bias", "relative_error": conv_bias_error},
        {"component": "conv2d_input", "relative_error": conv_input_error},
        {"component": "maxpool_input", "relative_error": pool_input_error},
        {"component": "global_average_pooling_input", "relative_error": gap_input_error},
    ]


def load_pixel_tensor(frame: pd.DataFrame, image_size: tuple[int, int], label: str) -> np.ndarray:
    rows = []
    for path in tqdm(frame["local_path"].tolist(), desc=label):
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
            array = np.asarray(rgb, dtype=np.float64) / 255.0
            rows.append(array.transpose(2, 0, 1))
    return np.stack(rows)


def channel_statistics(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = images.mean(axis=(0, 2, 3))
    std = images.std(axis=(0, 2, 3))
    std[std < 1e-12] = 1.0
    return mean, std


def normalize_pixels(images: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (images - mean[None, :, None, None]) / std[None, :, None, None]


def train_with_best_state(
    model: ManualCNNClassifier,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    validation_images: np.ndarray,
    validation_labels: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
) -> tuple[list[dict[str, float]], dict[str, np.ndarray], int]:
    history = []
    best_state = model.get_state()
    best_macro_f1 = -1.0
    best_epoch = 0
    for _ in range(epochs):
        epoch_history = model.fit(
            train_images,
            train_labels,
            validation_images=validation_images,
            validation_labels=validation_labels,
            epochs=1,
            batch_size=batch_size,
        )[0]
        epoch_history["epoch"] = float(len(history) + 1)
        history.append(epoch_history)
        if epoch_history["validation_macro_f1"] > best_macro_f1:
            best_macro_f1 = epoch_history["validation_macro_f1"]
            best_state = model.get_state()
            best_epoch = len(history)
    return history, best_state, best_epoch


def metadata_payload(
    config: dict[str, object],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
) -> dict[str, object]:
    return {
        "image_size": config["image_size"],
        "channel_mean": channel_mean.tolist(),
        "channel_std": channel_std.tolist(),
        "training_image_ids": train_frame["id"].astype(str).tolist(),
        "validation_image_ids": validation_frame["id"].astype(str).tolist(),
        "class_mapping": {"0": "true", "1": "fake"},
    }


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


def write_error_examples(predictions: pd.DataFrame, figures_dir: Path, per_group: int = 5) -> None:
    groups = {
        "correct_true": predictions[(predictions["2_way_label"] == 0) & (predictions["prediction"] == 0)].assign(
            confidence=1 - predictions["score_fake"]
        ),
        "correct_fake": predictions[(predictions["2_way_label"] == 1) & (predictions["prediction"] == 1)].assign(
            confidence=predictions["score_fake"]
        ),
        "false_positive": predictions[(predictions["2_way_label"] == 0) & (predictions["prediction"] == 1)].assign(
            confidence=predictions["score_fake"]
        ),
        "false_negative": predictions[(predictions["2_way_label"] == 1) & (predictions["prediction"] == 0)].assign(
            confidence=1 - predictions["score_fake"]
        ),
    }
    for name, group in groups.items():
        subset = group.sort_values("confidence", ascending=False).head(per_group)
        if subset.empty:
            continue
        fig, axes = plt.subplots(1, len(subset), figsize=(2.5 * len(subset), 2.8))
        axes = np.atleast_1d(axes)
        for axis, (_, row) in zip(axes, subset.iterrows(), strict=True):
            with Image.open(row["local_path"]) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(f"{row['id']}\n{row['score_fake']:.2f}", fontsize=8)
            axis.axis("off")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{name}.png", dpi=160)
        plt.close()


def comparison_table(cnn_metrics: dict[str, float], output_dir: Path) -> pd.DataFrame:
    rows = []
    for name, path in [
        ("Handcrafted LR", "reports/experiments/image_handcrafted_pilot/metrics.json"),
        ("Handcrafted MLP", "reports/experiments/image_mlp_pilot/metrics.json"),
    ]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append({"model": name, **select_metrics(data["validation_metrics"])})
    rows.append({"model": "CNN pixels only", **select_metrics(cnn_metrics)})
    frame = pd.DataFrame(rows)
    frame.to_markdown(output_dir / "comparison.md", index=False)
    return frame


def select_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "roc_auc": metrics["roc_auc"],
    }


def write_model_summary(path: Path, parameter_summary: list[dict[str, object]]) -> None:
    lines = ["Manual CNN parameter summary", ""]
    for row in parameter_summary:
        lines.append(f"{row['index']}: {row['layer']} - {row['parameters']} parameters")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(
    path: Path,
    metadata: dict[str, object],
    comparison: pd.DataFrame,
    confusion: np.ndarray,
) -> None:
    metrics = metadata["validation_metrics"]
    lines = [
        "# Manual CNN Image Pilot",
        "",
        "The CNN receives resized RGB pixels only. It does not receive original dimensions, file size, URL, subreddit, domain, or other metadata.",
        "",
        "Important: the 24-example tiny-overfit check did not reach high training accuracy. Gradient checks passed, so the convolution/backward code is locally consistent, but this CNN pilot should be treated as preliminary and under-optimized rather than as a final image model.",
        "",
        f"- Runtime seconds: {metadata['runtime_seconds']:.2f}",
        f"- Train size: {metadata['train_size']}",
        f"- Validation size: {metadata['validation_size']}",
        f"- Best validation epoch: {metadata['best_epoch']}",
        f"- Model size bytes: {metadata['model_size_bytes']}",
        "",
        "## Gradient Checks",
        "",
        pd.DataFrame(metadata["gradient_checks"]).to_markdown(index=False),
        "",
        "## Tiny Overfit Final Epoch",
        "",
        pd.DataFrame([metadata["tiny_overfit_final"]]).to_markdown(index=False),
        "",
        "## Validation Metrics",
        "",
        f"- Accuracy: {metrics['accuracy']:.6f}",
        f"- Precision: {metrics['precision']:.6f}",
        f"- Recall: {metrics['recall']:.6f}",
        f"- Macro F1: {metrics['macro_f1']:.6f}",
        f"- Weighted F1: {metrics['weighted_f1']:.6f}",
        f"- ROC-AUC: {metrics['roc_auc']:.6f}",
        "",
        "## Confusion Matrix",
        "",
        f"- TN: {confusion[0, 0]}",
        f"- FP: {confusion[0, 1]}",
        f"- FN: {confusion[1, 0]}",
        f"- TP: {confusion[1, 1]}",
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
    print(f"Wrote CNN pilot under {output_dir}")


if __name__ == "__main__":
    main()
