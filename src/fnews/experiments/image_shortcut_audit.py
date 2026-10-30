from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fnews.experiments.text_logreg_baseline import load_fakeddit_split
from fnews.ml.features.image_features import dense_rows_to_sparse
from fnews.ml.metrics.classification import binary_classification_metrics, confusion_matrix_binary
from fnews.ml.models.logistic_regression import ManualLogisticRegression
from fnews.ml.preprocessing.scaler import ManualStandardScaler

FEATURE_GROUPS = {
    "all_features": None,
    "metadata_surface": ["original_width", "original_height", "original_aspect_ratio"],
    "color_statistics": [
        "rgb_mean_r",
        "rgb_mean_g",
        "rgb_mean_b",
        "rgb_std_r",
        "rgb_std_g",
        "rgb_std_b",
        "brightness_mean",
        "brightness_std",
    ],
    "structural_edges": ["edge_mean", "edge_std"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit handcrafted image shortcuts and leakage.")
    parser.add_argument("--config", default="configs/experiments/image_shortcut_audit.json")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_audit(config: dict[str, object]) -> Path:
    started = time.perf_counter()
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_features = pd.read_parquet(str(config["train_features"]))
    validation_features = pd.read_parquet(str(config["validation_features"]))
    train_images = pd.read_parquet(str(config["train_images"]))
    validation_images = pd.read_parquet(str(config["validation_images"]))
    full_manifest = pd.read_parquet(str(config["image_manifest"]))

    feature_columns = [column for column in train_features.columns if column not in {"id", "2_way_label"}]
    feature_groups = build_feature_groups(feature_columns)
    coefficient_frame = feature_coefficients(
        train_features,
        feature_columns,
        Path(str(config["baseline_model_dir"])),
        Path(str(config["baseline_scaler_dir"])),
    )
    coefficient_frame.to_csv(output_dir / "feature_coefficients.csv", index=False)

    ablations = run_ablations(
        train_features,
        validation_features,
        feature_groups,
        seed=int(config["seed"]),
        model_config=config["model"],
    )
    ablations.to_csv(output_dir / "ablation_results.csv", index=False)

    permutations = run_permutation_sanity(
        train_features,
        validation_features,
        feature_groups["all_features"],
        seeds=[int(seed) for seed in config["permutation_seeds"]],
        model_config=config["model"],
    )
    permutations.to_csv(output_dir / "permutation_sanity.csv", index=False)

    label_distribution = label_distribution_report(
        full_manifest,
        train_images,
        validation_images,
        Path(str(config["full_train_tsv"])),
        Path(str(config["full_validation_tsv"])),
    )
    label_distribution.to_csv(output_dir / "label_distribution.csv", index=False)

    source_bias = source_bias_report(
        full_manifest,
        Path(str(config["full_train_tsv"])),
        Path(str(config["full_validation_tsv"])),
        train_images,
        validation_images,
    )
    source_bias.to_csv(output_dir / "source_availability_bias.csv", index=False)

    leakage = leakage_report(train_images, validation_images)
    leakage.to_csv(output_dir / "leakage_check.csv", index=False)

    metadata = {
        "config": config,
        "runtime_seconds": time.perf_counter() - started,
        "feature_count": len(feature_columns),
        "train_size": len(train_features),
        "validation_size": len(validation_features),
        "best_ablation_macro_f1": float(ablations["macro_f1"].max()),
        "metadata_only_macro_f1": metric_value(ablations, "metadata_surface", "macro_f1"),
        "color_statistics_macro_f1": metric_value(ablations, "color_statistics", "macro_f1"),
        "structural_edges_macro_f1": metric_value(ablations, "structural_edges", "macro_f1"),
        "shuffled_label_macro_f1_mean": float(permutations["macro_f1"].mean()),
        "leakage": leakage.to_dict(orient="records"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    Path(str(config["config_copy_path"])).parent.mkdir(parents=True, exist_ok=True)
    Path(str(config["config_copy_path"])).write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    write_readme(output_dir / "README.md", metadata, ablations, permutations, label_distribution, source_bias, leakage)
    return output_dir


def build_feature_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    groups = {}
    for name, columns in FEATURE_GROUPS.items():
        groups[name] = feature_columns if columns is None else [column for column in columns if column in feature_columns]
    histogram_columns = [column for column in feature_columns if column.startswith("hist_")]
    groups["color_statistics"] = groups["color_statistics"] + histogram_columns
    excluded = {"original_width", "original_height", "original_aspect_ratio", "file_size_bytes", "file_size_kb"}
    groups["full_without_surface_metadata"] = [column for column in feature_columns if column not in excluded]
    return groups


def feature_coefficients(
    train_features: pd.DataFrame,
    feature_columns: list[str],
    model_dir: Path,
    scaler_dir: Path,
) -> pd.DataFrame:
    model = ManualLogisticRegression.load(model_dir)
    scaler = ManualStandardScaler.load(scaler_dir)
    assert scaler.mean_ is not None
    assert scaler.scale_ is not None
    frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean": scaler.mean_,
            "std": scaler.scale_,
            "train_raw_mean": train_features[feature_columns].mean().to_numpy(),
            "train_raw_std": train_features[feature_columns].std(ddof=0).to_numpy(),
            "lr_coefficient": model.weights,
            "absolute_coefficient": np.abs(model.weights),
        }
    )
    frame = frame.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


def run_ablations(
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    *,
    seed: int,
    model_config: object,
) -> pd.DataFrame:
    rows = []
    for experiment, columns in feature_groups.items():
        result = train_evaluate_feature_subset(
            train_features,
            validation_features,
            columns,
            seed=seed,
            model_config=model_config,
        )
        rows.append({"experiment": experiment, "feature_count": len(columns), **result})
    return pd.DataFrame(rows).sort_values("experiment")


def run_permutation_sanity(
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    columns: list[str],
    *,
    seeds: list[int],
    model_config: object,
) -> pd.DataFrame:
    rows = []
    original_labels = train_features["2_way_label"].to_numpy(dtype=int)
    for seed in seeds:
        shuffled = train_features.copy()
        rng = np.random.default_rng(seed)
        shuffled["2_way_label"] = rng.permutation(original_labels)
        result = train_evaluate_feature_subset(
            shuffled,
            validation_features,
            columns,
            seed=seed,
            model_config=model_config,
        )
        rows.append({"experiment": "shuffled_train_labels", "seed": seed, "feature_count": len(columns), **result})
    return pd.DataFrame(rows)


def train_evaluate_feature_subset(
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    columns: list[str],
    *,
    seed: int,
    model_config: object,
) -> dict[str, float]:
    assert isinstance(model_config, dict)
    scaler = ManualStandardScaler()
    train_x = scaler.fit_transform(train_features[columns].to_numpy(dtype=np.float64))
    validation_x = scaler.transform(validation_features[columns].to_numpy(dtype=np.float64))
    train_vectors = dense_rows_to_sparse(train_x)
    validation_vectors = dense_rows_to_sparse(validation_x)
    train_y = train_features["2_way_label"].to_numpy(dtype=int)
    validation_y = validation_features["2_way_label"].to_numpy(dtype=int)
    model = ManualLogisticRegression(
        n_features=len(columns),
        learning_rate=float(model_config["learning_rate"]),
        l2=float(model_config["l2"]),
        seed=seed,
    )
    history = model.fit(
        train_vectors,
        train_y,
        validation_vectors=validation_vectors,
        validation_labels=validation_y,
        epochs=int(model_config["epochs"]),
        batch_size=int(model_config["batch_size"]),
        gradient_clip=float(model_config["gradient_clip"]),
        early_stopping_patience=int(model_config["early_stopping_patience"]),
    )
    scores = model.predict_proba(validation_vectors)
    predictions = (scores >= float(model_config["threshold"])).astype(int)
    metrics = binary_classification_metrics(validation_y, predictions, scores)
    confusion = confusion_matrix_binary(validation_y, predictions)
    return {
        "epochs_ran": float(len(history)),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "roc_auc": metrics["roc_auc"],
        "brier_score": metrics["brier_score"],
        "tn": float(confusion[0, 0]),
        "fp": float(confusion[0, 1]),
        "fn": float(confusion[1, 0]),
        "tp": float(confusion[1, 1]),
    }


def label_distribution_report(
    manifest: pd.DataFrame,
    train_images: pd.DataFrame,
    validation_images: pd.DataFrame,
    full_train_tsv: Path,
    full_validation_tsv: Path,
) -> pd.DataFrame:
    rows = []
    for split, frame in [
        ("downloaded_train", train_images),
        ("downloaded_validation", validation_images),
        ("full_multimodal_train", load_fakeddit_split(full_train_tsv)),
        ("full_multimodal_validation", load_fakeddit_split(full_validation_tsv)),
        ("manifest_downloaded_all", manifest[manifest["valid_image"]]),
        ("manifest_missing_or_invalid_all", manifest[~manifest["valid_image"]]),
    ]:
        counts = frame["2_way_label"].value_counts().sort_index()
        total = int(counts.sum())
        rows.append(
            {
                "group": split,
                "rows": total,
                "true_0": int(counts.get(0, 0)),
                "fake_1": int(counts.get(1, 0)),
                "true_pct": counts.get(0, 0) / total if total else 0.0,
                "fake_pct": counts.get(1, 0) / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def source_bias_report(
    manifest: pd.DataFrame,
    train_tsv: Path,
    validation_tsv: Path,
    train_images: pd.DataFrame,
    validation_images: pd.DataFrame,
) -> pd.DataFrame:
    train_meta = read_metadata(train_tsv, "train")
    validation_meta = read_metadata(validation_tsv, "validate")
    metadata = pd.concat([train_meta, validation_meta], ignore_index=True)
    downloaded_ids = set(train_images["id"].astype(str)) | set(validation_images["id"].astype(str))
    metadata["downloaded_valid_image"] = metadata["id"].astype(str).isin(downloaded_ids)
    rows = []
    for column in ["domain", "subreddit"]:
        rows.extend(top_bias_rows(metadata, column))
    rows.extend(label_download_bias_rows(manifest))
    return pd.DataFrame(rows)


def read_metadata(path: Path, split: str) -> pd.DataFrame:
    columns = ["id", "2_way_label", "domain", "subreddit", "created_utc"]
    frame = pd.read_csv(path, sep="\t", usecols=columns, low_memory=False)
    frame["split"] = split
    return frame


def top_bias_rows(metadata: pd.DataFrame, column: str, top_n: int = 15) -> list[dict[str, object]]:
    rows = []
    for downloaded_value, group_name in [(True, "downloaded_valid"), (False, "not_downloaded_or_not_in_pilot")]:
        group = metadata[metadata["downloaded_valid_image"] == downloaded_value]
        counts = group[column].astype("string").fillna("<missing>").value_counts().head(top_n)
        for value, count in counts.items():
            rows.append(
                {
                    "analysis": f"top_{column}",
                    "group": group_name,
                    "value": str(value),
                    "count": int(count),
                    "share_within_group": float(count / len(group)) if len(group) else 0.0,
                }
            )
    return rows


def label_download_bias_rows(manifest: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for split, group in manifest.groupby("split", observed=True):
        for label, label_group in group.groupby("2_way_label"):
            rows.append(
                {
                    "analysis": "download_rate_by_label",
                    "group": str(split),
                    "value": str(label),
                    "count": int(label_group["valid_image"].sum()),
                    "share_within_group": float(label_group["valid_image"].mean()),
                }
            )
    return rows


def leakage_report(train_images: pd.DataFrame, validation_images: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(overlap_row("id", train_images, validation_images))
    rows.append(overlap_row("image_url", train_images, validation_images))
    rows.append(overlap_row("sha256", train_images, validation_images))
    rows.append(overlap_row("perceptual_hash", train_images, validation_images))
    return pd.DataFrame(rows)


def overlap_row(column: str, train_images: pd.DataFrame, validation_images: pd.DataFrame) -> dict[str, object]:
    if column not in train_images.columns or column not in validation_images.columns:
        return {"field": column, "train_unique": 0, "validation_unique": 0, "overlap_count": None}
    train_values = set(train_images[column].dropna().astype(str))
    validation_values = set(validation_images[column].dropna().astype(str))
    return {
        "field": column,
        "train_unique": len(train_values),
        "validation_unique": len(validation_values),
        "overlap_count": len(train_values & validation_values),
    }


def metric_value(frame: pd.DataFrame, experiment: str, metric: str) -> float:
    match = frame.loc[frame["experiment"] == experiment, metric]
    return float(match.iloc[0]) if len(match) else float("nan")


def write_readme(
    path: Path,
    metadata: dict[str, object],
    ablations: pd.DataFrame,
    permutations: pd.DataFrame,
    label_distribution: pd.DataFrame,
    source_bias: pd.DataFrame,
    leakage: pd.DataFrame,
) -> None:
    top_source = source_bias.head(20)
    lines = [
        "# Image Shortcut Audit",
        "",
        "This audit checks whether the handcrafted image baseline is driven by visual statistics, surface metadata, sample bias, or duplicate leakage.",
        "",
        f"- Runtime seconds: {metadata['runtime_seconds']:.2f}",
        f"- Feature count audited: {metadata['feature_count']}",
        f"- Train images: {metadata['train_size']}",
        f"- Validation images: {metadata['validation_size']}",
        "",
        "## Ablation Results",
        "",
        ablations.to_markdown(index=False),
        "",
        "## Shuffled-Label Sanity",
        "",
        permutations.to_markdown(index=False),
        "",
        "## Label Distribution",
        "",
        label_distribution.to_markdown(index=False),
        "",
        "## Leakage Check",
        "",
        leakage.to_markdown(index=False),
        "",
        "## Source / Availability Bias Snapshot",
        "",
        top_source.to_markdown(index=False),
        "",
        "## Conclusion",
        "",
        f"- Metadata-only Macro F1: {metadata['metadata_only_macro_f1']:.6f}",
        f"- Color/statistics Macro F1: {metadata['color_statistics_macro_f1']:.6f}",
        f"- Structural/edge Macro F1: {metadata['structural_edges_macro_f1']:.6f}",
        f"- Shuffled-label Macro F1 mean: {metadata['shuffled_label_macro_f1_mean']:.6f}",
        "- Do not interpret image performance as fake-news understanding. This audit only measures predictive shortcuts available in the pilot images.",
        "- CNN comparisons should use pixels only, excluding original dimensions, file size, URL, subreddit, domain, and other metadata.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = run_audit(load_config(args.config))
    print(f"Wrote image shortcut audit under {output_dir}")


if __name__ == "__main__":
    main()
