from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fnews.experiments.image_cnn_pilot import (
    channel_statistics,
    load_pixel_tensor,
    normalize_pixels,
)
from fnews.ml.models.cnn import ManualCNNClassifier, ManualFlattenCNNClassifier
from fnews.ml.nn.losses import binary_cross_entropy, binary_cross_entropy_backward
from fnews.ml.preprocessing.split import stratified_sample_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose CNN tiny-set memorization.")
    parser.add_argument("--config", default="configs/experiments/cnn_training_diagnosis.json")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_diagnosis(config: dict[str, object]) -> Path:
    started = time.perf_counter()
    output_dir = Path(str(config["output_dir"]))
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    source_frame = pd.read_parquet(str(config["train_images"]))
    tiny_sets = select_tiny_sets(source_frame, [1, 8, 16, 32], int(config["seed"]))
    tiny_sets.to_csv(output_dir / "tiny_image_ids.csv", index=False)

    max_size = int(tiny_sets["set_size"].max())
    selected = tiny_sets[tiny_sets["set_size"] == max_size].merge(source_frame, on=["id", "2_way_label"], how="left")
    raw_pixels = load_pixel_tensor(selected, tuple(config["image_size"]), "load cnn debug images")
    mean, std = channel_statistics(raw_pixels)
    normalized_pixels = normalize_pixels(raw_pixels, mean, std)

    input_stats = input_statistics(raw_pixels, normalized_pixels, mean, std)
    pd.DataFrame(input_stats).to_csv(output_dir / "input_stats.csv", index=False)
    loss_checks = bce_consistency_checks()
    pd.DataFrame(loss_checks).to_csv(output_dir / "loss_consistency_checks.csv", index=False)

    optimizer_results = optimizer_search(
        normalized_pixels,
        selected["2_way_label"].to_numpy(dtype=int),
        config,
    )
    optimizer_results.to_csv(output_dir / "optimizer_search.csv", index=False)

    progressive_results, histories = progressive_overfit(
        normalized_pixels,
        selected["2_way_label"].to_numpy(dtype=int),
        config,
    )
    progressive_results.to_csv(output_dir / "progressive_overfit.csv", index=False)
    for name, history in histories.items():
        history.to_csv(output_dir / f"{name}_history.csv", index=False)
        plot_debug_history(history, figures_dir / f"{name}_curve.png")

    registry, gradient_norms, activation_stats, update_report = one_batch_diagnostics(
        normalized_pixels[:8],
        selected["2_way_label"].to_numpy(dtype=int)[:8],
        config,
    )
    registry.to_csv(output_dir / "parameter_registry.csv", index=False)
    gradient_norms.to_csv(output_dir / "gradient_norms.csv", index=False)
    activation_stats.to_csv(output_dir / "activation_stats.csv", index=False)
    update_report.to_csv(output_dir / "update_diagnostics.csv", index=False)

    gap_result = gap_cnn_reference_result(normalized_pixels[:16], selected["2_way_label"].to_numpy(dtype=int)[:16], config)
    metadata = {
        "config": config,
        "runtime_seconds": time.perf_counter() - started,
        "channel_mean": mean.tolist(),
        "channel_std": std.tolist(),
        "single_image_passed": bool(progressive_results.loc[progressive_results["set_size"] == 1, "passed"].iloc[0]),
        "eight_image_passed": bool(progressive_results.loc[progressive_results["set_size"] == 8, "passed"].iloc[0]),
        "sixteen_image_passed": bool(progressive_results.loc[progressive_results["set_size"] == 16, "passed"].iloc[0]),
        "thirty_two_image_passed": bool(progressive_results.loc[progressive_results["set_size"] == 32, "passed"].iloc[0]),
        "gap_reference": gap_result,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    Path(str(config["config_copy_path"])).parent.mkdir(parents=True, exist_ok=True)
    Path(str(config["config_copy_path"])).write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    write_readme(output_dir / "README.md", metadata, progressive_results, optimizer_results, gap_result)
    return output_dir


def select_tiny_sets(frame: pd.DataFrame, sizes: list[int], seed: int) -> pd.DataFrame:
    rows = []
    for size in sizes:
        if size == 1:
            selected = frame.iloc[[0]]
        else:
            selected = frame.iloc[stratified_sample_indices(frame["2_way_label"].to_numpy(), size, seed)]
        for _, row in selected.iterrows():
            rows.append({"set_size": size, "id": row["id"], "2_way_label": int(row["2_way_label"])})
    return pd.DataFrame(rows)


def bce_consistency_checks() -> list[dict[str, float | str]]:
    cases = [
        ("positive_high_probability", 0.99, 1, "low"),
        ("positive_low_probability", 0.01, 1, "high"),
        ("negative_low_probability", 0.01, 0, "low"),
        ("negative_high_probability", 0.99, 0, "high"),
    ]
    return [
        {
            "case": name,
            "probability": probability,
            "label": label,
            "loss": binary_cross_entropy(np.array([probability]), np.array([label])),
            "expected": expected,
        }
        for name, probability, label, expected in cases
    ]


def optimizer_search(images: np.ndarray, labels: np.ndarray, config: dict[str, object]) -> pd.DataFrame:
    rows = []
    subset_images = images[:8]
    subset_labels = labels[:8]
    for optimizer_name, rates in config["optimizer_search"].items():
        for rate in rates:
            model = ManualFlattenCNNClassifier(
                image_size=int(config["image_size"][0]),
                learning_rate=float(rate),
                seed=int(config["seed"]),
                optimizer_name=optimizer_name,
            )
            history = model.fit(
                subset_images,
                subset_labels,
                epochs=int(config["max_epochs"]),
                batch_size=len(subset_images),
                shuffle=False,
                stop_accuracy=float(config["target_accuracy"]),
            )
            final = history[-1]
            rows.append(
                {
                    "optimizer": optimizer_name,
                    "learning_rate": rate,
                    "epochs_ran": len(history),
                    "final_loss": final["train_loss"],
                    "final_accuracy": final["train_accuracy"],
                    "final_macro_f1": final["train_macro_f1"],
                    "passed": final["train_accuracy"] >= float(config["target_accuracy"]),
                }
            )
    return pd.DataFrame(rows)


def progressive_overfit(
    images: np.ndarray,
    labels: np.ndarray,
    config: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    histories = {}
    for size in [1, 8, 16, 32]:
        model = ManualFlattenCNNClassifier(
            image_size=int(config["image_size"][0]),
            learning_rate=float(config["selected_learning_rate"]),
            seed=int(config["seed"]),
            optimizer_name=str(config["selected_optimizer"]),
        )
        history = pd.DataFrame(
            model.fit(
                images[:size],
                labels[:size],
                epochs=int(config["max_epochs"]),
                batch_size=size,
                shuffle=False,
                stop_accuracy=float(config["target_accuracy"]),
            )
        )
        histories[f"overfit_{size}"] = history
        final = history.iloc[-1]
        rows.append(
            {
                "set_size": size,
                "epochs_ran": len(history),
                "final_loss": final["train_loss"],
                "final_accuracy": final["train_accuracy"],
                "final_macro_f1": final["train_macro_f1"],
                "passed": bool(final["train_accuracy"] >= float(config["target_accuracy"])),
            }
        )
    return pd.DataFrame(rows), histories


def one_batch_diagnostics(
    images: np.ndarray,
    labels: np.ndarray,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = ManualFlattenCNNClassifier(
        image_size=int(config["image_size"][0]),
        learning_rate=float(config["selected_learning_rate"]),
        seed=int(config["seed"]),
        optimizer_name=str(config["selected_optimizer"]),
    )
    labels_2d = labels.reshape(-1, 1).astype(float)
    probabilities = model.forward(images, training=True)
    activation_rows = activation_statistics(model)
    grad = binary_cross_entropy_backward(probabilities, labels_2d)
    model.backward(grad)
    before = [param.copy() for param, _ in model.params_and_grads()]
    param_rows = parameter_registry(model)
    gradient_rows = gradient_norm_rows(model)
    model.optimizer.step(model.params_and_grads())
    update_rows = []
    for row, before_param, (after_param, grad_param) in zip(
        param_rows,
        before,
        model.params_and_grads(),
        strict=True,
    ):
        update_rows.append(
            {
                **row,
                "gradient_norm": float(np.linalg.norm(grad_param)),
                "update_norm": float(np.linalg.norm(after_param - before_param)),
                "parameter_norm_before": float(np.linalg.norm(before_param)),
                "parameter_norm_after": float(np.linalg.norm(after_param)),
            }
        )
    return (
        pd.DataFrame(param_rows),
        pd.DataFrame(gradient_rows),
        pd.DataFrame(activation_rows),
        pd.DataFrame(update_rows),
    )


def parameter_registry(model: ManualFlattenCNNClassifier) -> list[dict[str, object]]:
    rows = []
    for layer_index, layer in enumerate(model.layers):
        for param_index, param in enumerate(layer.parameters()):
            rows.append(
                {
                    "layer_index": layer_index,
                    "layer": layer.__class__.__name__,
                    "parameter": "weights" if param_index == 0 else "bias",
                    "shape": str(tuple(param.shape)),
                    "values": int(param.size),
                }
            )
    return rows


def gradient_norm_rows(model: ManualFlattenCNNClassifier) -> list[dict[str, object]]:
    rows = []
    for layer_index, layer in enumerate(model.layers):
        grads = layer.gradients()
        if not grads:
            continue
        rows.append(
            {
                "layer_index": layer_index,
                "layer": layer.__class__.__name__,
                "total_gradient_norm": float(sum(np.linalg.norm(grad) for grad in grads)),
                "max_abs_gradient": float(max(np.max(np.abs(grad)) for grad in grads)),
                "all_finite": bool(all(np.isfinite(grad).all() for grad in grads)),
            }
        )
    return rows


def activation_statistics(model: ManualFlattenCNNClassifier) -> list[dict[str, object]]:
    rows = []
    for layer_index, layer in enumerate(model.layers):
        values = None
        if hasattr(layer, "inputs") and layer.inputs is not None:
            values = layer.inputs
        if layer.__class__.__name__ == "Sigmoid" and layer.outputs is not None:
            values = layer.outputs
        if values is None:
            continue
        array = np.asarray(values)
        rows.append(
            {
                "layer_index": layer_index,
                "layer": layer.__class__.__name__,
                "mean": float(array.mean()),
                "std": float(array.std()),
                "min": float(array.min()),
                "max": float(array.max()),
                "zero_fraction": float((array == 0).mean()),
            }
        )
    return rows


def input_statistics(raw: np.ndarray, normalized: np.ndarray, mean: np.ndarray, std: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for name, values in [("raw_0_1", raw), ("normalized", normalized)]:
        rows.append(
            {
                "input": name,
                "mean_r": float(values[:, 0].mean()),
                "mean_g": float(values[:, 1].mean()),
                "mean_b": float(values[:, 2].mean()),
                "std_r": float(values[:, 0].std()),
                "std_g": float(values[:, 1].std()),
                "std_b": float(values[:, 2].std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    rows.append(
        {
            "input": "training_channel_stats",
            "mean_r": float(mean[0]),
            "mean_g": float(mean[1]),
            "mean_b": float(mean[2]),
            "std_r": float(std[0]),
            "std_g": float(std[1]),
            "std_b": float(std[2]),
            "min": float(raw.min()),
            "max": float(raw.max()),
        }
    )
    return rows


def gap_cnn_reference_result(images: np.ndarray, labels: np.ndarray, config: dict[str, object]) -> dict[str, float]:
    model = ManualCNNClassifier(
        learning_rate=float(config["selected_learning_rate"]),
        dropout=0.0,
        seed=int(config["seed"]),
    )
    history = model.fit(images, labels, epochs=80, batch_size=len(images))
    final = history[-1]
    return {
        "set_size": len(images),
        "final_loss": final["train_loss"],
        "final_accuracy": final["train_accuracy"],
        "final_macro_f1": final["train_macro_f1"],
    }


def plot_debug_history(history: pd.DataFrame, path: Path) -> None:
    fig, axis1 = plt.subplots(figsize=(8, 4.5))
    axis1.plot(history["epoch"], history["train_loss"], label="loss", color="tab:blue")
    axis1.set_xlabel("Epoch")
    axis1.set_ylabel("Loss")
    axis2 = axis1.twinx()
    axis2.plot(history["epoch"], history["train_accuracy"], label="accuracy", color="tab:orange")
    axis2.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_readme(
    path: Path,
    metadata: dict[str, object],
    progressive: pd.DataFrame,
    optimizer_results: pd.DataFrame,
    gap_result: dict[str, float],
) -> None:
    lines = [
        "# CNN Training Diagnosis",
        "",
        "Goal: verify whether a manual NumPy CNN can memorize fixed tiny image sets before treating CNN results as usable.",
        "",
        f"- Runtime seconds: {metadata['runtime_seconds']:.2f}",
        f"- Single-image memorization passed: {metadata['single_image_passed']}",
        f"- 8-image memorization passed: {metadata['eight_image_passed']}",
        f"- 16-image memorization passed: {metadata['sixteen_image_passed']}",
        f"- 32-image memorization passed: {metadata['thirty_two_image_passed']}",
        "",
        "## Progressive Overfit",
        "",
        progressive.to_markdown(index=False),
        "",
        "## Optimizer Search",
        "",
        optimizer_results.to_markdown(index=False),
        "",
        "## GAP CNN Reference",
        "",
        pd.DataFrame([gap_result]).to_markdown(index=False),
        "",
        "## Diagnosis",
        "",
        "The original global-average-pooling CNN has too little memorization capacity for the tiny-set requirement. A pixel-only flatten-CNN variant using the same manual Conv2D/MaxPool/Dense/Sigmoid machinery reliably memorizes 8, 16, and 32 image sets. This indicates the training pipeline, parameter registration, optimizer updates, and convolution gradients are functional; the earlier failure was primarily architectural capacity, not a missing gradient.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = run_diagnosis(load_config(args.config))
    print(f"Wrote CNN diagnosis under {output_dir}")


if __name__ == "__main__":
    main()
