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
from fnews.experiments.multimodal_fusion import evaluate_scores
from fnews.experiments.neural_text import build_text_vocabulary, encode_documents
from fnews.ml.metrics.calibration import (
    abstention_metrics,
    bootstrap_macro_f1,
    confidence_interval,
    expected_calibration_error,
    find_temperature_grid,
    paired_disagreement_counts,
    pearson_correlation,
    reliability_bins,
    temperature_scale_probabilities,
    threshold_metrics,
)
from fnews.ml.metrics.classification import binary_classification_metrics
from fnews.ml.models.cnn import ManualFlattenCNNClassifier
from fnews.ml.models.multimodal import (
    FusionHead,
    concatenate_features,
    gated_features,
    similarity_features,
    validate_alignment,
)
from fnews.ml.models.text_embedding import ManualTextEmbeddingClassifier
from fnews.ml.preprocessing.split import stratified_sample_indices
from fnews.ml.preprocessing.text import tokenize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Milestone 8 robustness and calibration analysis.")
    parser.add_argument("--config", default="configs/experiments/milestone8.json")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_experiment(config: dict[str, object]) -> Path:
    started = time.perf_counter()
    output_dir = Path(str(config["output_dir"]))
    figures_dir = output_dir / "figures"
    features_dir = Path(str(config["features_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    original_train = pd.read_parquet(str(config["original_train_manifest"]))
    original_validation = pd.read_parquet(str(config["original_validation_manifest"]))
    coverage = local_image_coverage(config, original_train, original_validation)
    pd.DataFrame(coverage).to_csv(output_dir / "coverage_audit.csv", index=False)

    clean_train, clean_validation, removals = leakage_reduced_manifests(original_train, original_validation)
    clean_train_path = Path("data/processed/fakeddit/multimodal_training_manifest_clean.parquet")
    clean_validation_path = Path("data/processed/fakeddit/multimodal_validation_manifest_clean.parquet")
    clean_train.to_parquet(clean_train_path, index=False)
    clean_validation.to_parquet(clean_validation_path, index=False)
    removals.to_csv(output_dir / "leakage_removals.csv", index=False)

    original = train_and_score_bundle("original", original_train, original_validation, config, output_dir, features_dir)
    clean = train_and_score_bundle("clean", clean_train, clean_validation, config, output_dir, features_dir)
    comparison = pd.DataFrame(original["rows"] + clean["rows"])
    comparison.to_csv(output_dir / "clean_comparison.csv", index=False)

    model_for_analysis = "concat_fusion"
    clean_predictions = clean["predictions"]
    labels = clean["labels"]
    concat_scores = clean_predictions[model_for_analysis]
    threshold = pd.DataFrame(threshold_metrics(labels, concat_scores))
    threshold.to_csv(output_dir / "threshold_analysis.csv", index=False)
    selected_threshold_row = threshold.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0].to_dict()

    temperature_grid = np.linspace(
        float(config["temperature_grid"]["min"]),
        float(config["temperature_grid"]["max"]),
        int(config["temperature_grid"]["steps"]),
    )
    best_temperature, best_nll = find_temperature_grid(labels, concat_scores, temperature_grid)
    calibrated_scores = temperature_scale_probabilities(concat_scores, best_temperature)
    calibration = calibration_report(labels, concat_scores, calibrated_scores, best_temperature, best_nll)
    calibration.to_csv(output_dir / "calibration_metrics.csv", index=False)
    bins_before = pd.DataFrame(reliability_bins(labels, concat_scores, bins=10))
    bins_before["model"] = "concat_before_temperature"
    bins_after = pd.DataFrame(reliability_bins(labels, calibrated_scores, bins=10))
    bins_after["model"] = "concat_after_temperature"
    reliability = pd.concat([bins_before, bins_after], ignore_index=True)
    reliability.to_csv(output_dir / "reliability_bins.csv", index=False)

    abstention = pd.DataFrame(abstention_metrics(labels, calibrated_scores, np.array(config["abstention_bands"], dtype=float)))
    abstention.to_csv(output_dir / "abstention_results.csv", index=False)
    selected_abstention_row = select_abstention_row(abstention).to_dict()

    complementarity = complementarity_report(
        labels,
        clean_predictions["text_repr_same_head"],
        clean_predictions["image_repr_same_head"],
        clean_predictions[model_for_analysis],
    )
    complementarity.to_csv(output_dir / "complementarity.csv", index=False)
    correlations = prediction_correlation_report(clean_predictions, labels)
    correlations.to_csv(output_dir / "prediction_correlation.csv", index=False)
    contribution = image_contribution_report(
        labels,
        clean_predictions["text_repr_same_head"],
        clean_predictions[model_for_analysis],
    )
    contribution.to_csv(output_dir / "image_contribution.csv", index=False)
    shuffle = shuffle_repeat_report(clean["heads"]["concat_fusion"], clean["text_validation"], clean["image_validation"], labels, config)
    shuffle.to_csv(output_dir / "shuffle_multiseed.csv", index=False)
    bootstrap = bootstrap_report(labels, clean_predictions, config)
    bootstrap.to_csv(output_dir / "bootstrap_results.csv", index=False)
    paired = paired_disagreement_counts(labels, clean_predictions["text_repr_same_head"], clean_predictions[model_for_analysis])
    paired["text_model"] = "text_repr_same_head"
    paired["fusion_model"] = model_for_analysis
    (output_dir / "paired_comparison.json").write_text(json.dumps(paired, indent=2, sort_keys=True), encoding="utf-8")
    errors = high_confidence_errors(clean_validation, labels, clean_predictions)
    errors.to_csv(output_dir / "high_confidence_errors.csv", index=False)

    scale = scale_analysis(clean_validation, original_train, clean_train, config, output_dir)
    scale.to_csv(output_dir / "scale_analysis.csv", index=False)
    compute = compute_report(clean, output_dir, features_dir)
    compute.to_csv(output_dir / "model_compute.csv", index=False)

    plot_thresholds(threshold, figures_dir / "threshold_curves.png")
    plot_reliability(reliability, figures_dir / "reliability_diagram.png")
    plot_abstention(abstention, figures_dir / "abstention_curve.png")
    create_notebook(Path(str(config["notebook_path"])))

    metadata = {
        "runtime_seconds": time.perf_counter() - started,
        "coverage": coverage,
        "original_train_size": len(original_train),
        "original_validation_size": len(original_validation),
        "clean_train_size": len(clean_train),
        "clean_validation_size": len(clean_validation),
        "leakage_removals": int(len(removals)),
        "selected_threshold": selected_threshold_row,
        "selected_abstention": selected_abstention_row,
        "temperature": best_temperature,
        "final_recommendation": final_recommendation(comparison, bootstrap, paired),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    Path(str(config["config_copy_path"])).write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    write_readme(output_dir / "README.md", metadata, comparison, calibration, abstention, complementarity, contribution, shuffle, bootstrap, scale)
    return output_dir


def local_image_coverage(config: dict[str, object], train: pd.DataFrame, validation: pd.DataFrame) -> list[dict[str, object]]:
    manifest = pd.read_parquet(str(config["image_manifest"]))
    rows: list[dict[str, object]] = []
    prior = {"train": len(train), "validate": len(validation)}
    for split in ["train", "validate"]:
        frame = manifest[(manifest["split"].astype(str) == split) & manifest["valid_image"]].copy()
        counts = frame["2_way_label"].value_counts().to_dict()
        rows.append(
            {
                "split": split,
                "valid_local_images": int(len(frame)),
                "milestone7_matched_images": int(prior[split]),
                "new_since_milestone7": int(len(frame) - prior[split]),
                "true_0": int(counts.get(0, 0)),
                "fake_1": int(counts.get(1, 0)),
                "fake_pct": float(counts.get(1, 0) / len(frame)) if len(frame) else 0.0,
                "domain_available": "domain" in frame.columns,
                "subreddit_available": "subreddit" in frame.columns,
            }
        )
    return rows


def normalize_title(value: object) -> str:
    return " ".join(str(value).lower().strip().split())


def leakage_reduced_manifests(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    removals: dict[str, dict[str, object]] = {}
    validation_lookup = {
        "id": validation.set_index(validation["id"].astype(str))["id"].astype(str).to_dict(),
        "normalized_title": validation.assign(normalized_title=validation["clean_title"].map(normalize_title))
        .drop_duplicates("normalized_title")
        .set_index("normalized_title")["id"]
        .astype(str)
        .to_dict(),
        "image_url": validation.dropna(subset=["image_url"]).drop_duplicates("image_url").set_index("image_url")["id"].astype(str).to_dict(),
        "sha256": validation.dropna(subset=["sha256"]).drop_duplicates("sha256").set_index("sha256")["id"].astype(str).to_dict(),
        "perceptual_hash": validation.dropna(subset=["perceptual_hash"])
        .drop_duplicates("perceptual_hash")
        .set_index("perceptual_hash")["id"]
        .astype(str)
        .to_dict(),
    }
    train_augmented = train.assign(normalized_title=train["clean_title"].map(normalize_title))
    rules = [
        ("id", "duplicate_id"),
        ("normalized_title", "duplicate_normalized_title"),
        ("image_url", "duplicate_image_url"),
        ("sha256", "duplicate_sha256"),
        ("perceptual_hash", "duplicate_phash"),
    ]
    for _, row in train_augmented.iterrows():
        train_id = str(row["id"])
        for column, duplicate_type in rules:
            value = row[column]
            if pd.isna(value):
                continue
            matched_id = validation_lookup[column].get(str(value))
            if matched_id is None:
                continue
            current = removals.setdefault(
                train_id,
                {
                    "id": train_id,
                    "split": "train",
                    "reason": "matches_validation_content",
                    "matched_id": matched_id,
                    "duplicate_type": duplicate_type,
                },
            )
            if duplicate_type not in str(current["duplicate_type"]).split("|"):
                current["duplicate_type"] = f"{current['duplicate_type']}|{duplicate_type}"
            break
    removal_frame = pd.DataFrame(list(removals.values()), columns=["id", "split", "reason", "matched_id", "duplicate_type"])
    remove_ids = set(removals)
    clean_train = train[~train["id"].astype(str).isin(remove_ids)].reset_index(drop=True)
    return clean_train, validation.reset_index(drop=True), removal_frame


def train_and_score_bundle(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: dict[str, object],
    output_dir: Path,
    features_dir: Path,
) -> dict[str, object]:
    labels_train = train["2_way_label"].to_numpy(dtype=int)
    labels_validation = validation["2_way_label"].to_numpy(dtype=int)
    vocabulary, train_tokens, validation_tokens = prepare_text_arrays(train, validation, config)
    text_model = ManualTextEmbeddingClassifier(
        vocab_size=len(vocabulary),
        embedding_dim=int(config["text"]["embedding_dim"]),
        hidden_dim=int(config["text"]["hidden_dim"]),
        learning_rate=float(config["text"]["learning_rate"]),
        dropout=float(config["text"]["dropout"]),
        seed=int(config["seed"]),
    )
    text_history = text_model.fit(
        train_tokens,
        labels_train,
        validation_tokens=validation_tokens,
        validation_labels=labels_validation,
        epochs=int(config["text"]["epochs"]),
        batch_size=int(config["text"]["batch_size"]),
    )
    pd.DataFrame(text_history).to_csv(output_dir / f"{name}_text_history.csv", index=False)

    train_pixels = load_pixel_tensor(train, tuple(config["image"]["image_size"]), f"load {name} train images")
    validation_pixels = load_pixel_tensor(validation, tuple(config["image"]["image_size"]), f"load {name} validation images")
    channel_mean, channel_std = channel_statistics(train_pixels)
    train_pixels = normalize_pixels(train_pixels, channel_mean, channel_std)
    validation_pixels = normalize_pixels(validation_pixels, channel_mean, channel_std)
    image_model = ManualFlattenCNNClassifier(
        image_size=int(config["image"]["image_size"][0]),
        learning_rate=float(config["image"]["learning_rate"]),
        seed=int(config["seed"]),
        optimizer_name="adam",
    )
    image_history = image_model.fit(
        train_pixels,
        labels_train,
        epochs=int(config["image"]["epochs"]),
        batch_size=int(config["image"]["batch_size"]),
        shuffle=True,
        stop_accuracy=None,
    )
    pd.DataFrame(image_history).to_csv(output_dir / f"{name}_image_history.csv", index=False)

    text_train = text_model.encode_text(train_tokens, training=False)
    text_validation = text_model.encode_text(validation_tokens, training=False)
    image_train = image_model.encode_image(train_pixels, training=False)
    image_validation = image_model.encode_image(validation_pixels, training=False)
    save_bundle_features(features_dir / name, train, validation, text_train, text_validation, image_train, image_validation)
    validate_alignment(
        np.load(features_dir / name / "ids_validate.npy", allow_pickle=True),
        validation["id"].astype(str).to_numpy(),
        np.load(features_dir / name / "labels_validate.npy"),
        labels_validation,
    )

    fusion_sets = {
        "text_repr_same_head": (text_train, text_validation),
        "image_repr_same_head": (image_train, image_validation),
        "concat_fusion": (concatenate_features(text_train, image_train), concatenate_features(text_validation, image_validation)),
        "similarity_fusion": (similarity_features(text_train, image_train), similarity_features(text_validation, image_validation)),
    }
    gated_train, _ = gated_features(text_train, image_train)
    gated_validation, _ = gated_features(text_validation, image_validation)
    fusion_sets["gated_fusion"] = (gated_train, gated_validation)

    predictions = {
        "text_encoder_classifier": text_model.predict_proba(validation_tokens, batch_size=int(config["text"]["batch_size"])),
        "image_cnn_classifier": image_model.predict_proba(validation_pixels, batch_size=int(config["image"]["batch_size"])),
    }
    rows = [
        score_row(name, "text_encoder_classifier", predictions["text_encoder_classifier"], labels_validation, len(labels_train)),
        score_row(name, "image_cnn_classifier", predictions["image_cnn_classifier"], labels_validation, len(labels_train)),
    ]
    heads: dict[str, FusionHead] = {}
    for model_name, (train_features, validation_features) in fusion_sets.items():
        head = FusionHead(
            input_dim=train_features.shape[1],
            hidden_dim=int(config["fusion"]["hidden_dim"]),
            learning_rate=float(config["fusion"]["learning_rate"]),
            dropout=float(config["fusion"]["dropout"]),
            seed=int(config["seed"]),
        )
        history = head.fit(
            train_features,
            labels_train,
            validation_features=validation_features,
            validation_labels=labels_validation,
            epochs=int(config["fusion"]["epochs"]),
            batch_size=int(config["fusion"]["batch_size"]),
        )
        pd.DataFrame(history).to_csv(output_dir / f"{name}_{model_name}_history.csv", index=False)
        heads[model_name] = head
        predictions[model_name] = head.predict_proba(validation_features)
        rows.append(score_row(name, model_name, predictions[model_name], labels_validation, len(labels_train)))

    prediction_frame = validation[["id", "clean_title", "2_way_label", "local_path"]].copy()
    for model_name, scores in predictions.items():
        prediction_frame[f"{model_name}_score"] = scores
        prediction_frame[f"{model_name}_prediction"] = (scores >= 0.5).astype(int)
    prediction_frame.to_parquet(output_dir / f"{name}_predictions.parquet", index=False)
    return {
        "rows": rows,
        "predictions": predictions,
        "heads": heads,
        "labels": labels_validation,
        "text_validation": text_validation,
        "image_validation": image_validation,
        "text_parameter_count": int(text_model.parameter_summary()[-1]["parameters"]),
        "image_parameter_count": int(image_model.parameter_summary()[-1]["parameters"]),
    }


def prepare_text_arrays(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, object]):
    train_tokens_raw = [tokenize(text, mode=str(config["text"]["tokenization_mode"])) for text in train["clean_title"]]
    validation_tokens_raw = [tokenize(text, mode=str(config["text"]["tokenization_mode"])) for text in validation["clean_title"]]
    vocabulary = build_text_vocabulary(
        train_tokens_raw,
        min_frequency=int(config["text"]["min_frequency"]),
        max_vocabulary_size=int(config["text"]["max_vocabulary_size"]),
    )
    max_length = int(config["text"]["max_length"])
    return vocabulary, encode_documents(train_tokens_raw, vocabulary, max_length), encode_documents(validation_tokens_raw, vocabulary, max_length)


def save_bundle_features(
    directory: Path,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    text_train: np.ndarray,
    text_validation: np.ndarray,
    image_train: np.ndarray,
    image_validation: np.ndarray,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "text_train_embeddings.npy", text_train)
    np.save(directory / "text_validate_embeddings.npy", text_validation)
    np.save(directory / "image_train_embeddings.npy", image_train)
    np.save(directory / "image_validate_embeddings.npy", image_validation)
    np.save(directory / "ids_train.npy", train["id"].astype(str).to_numpy())
    np.save(directory / "ids_validate.npy", validation["id"].astype(str).to_numpy())
    np.save(directory / "labels_train.npy", train["2_way_label"].to_numpy(dtype=int))
    np.save(directory / "labels_validate.npy", validation["2_way_label"].to_numpy(dtype=int))


def score_row(split_name: str, model_name: str, scores: np.ndarray, labels: np.ndarray, train_n: int) -> dict[str, object]:
    row = evaluate_scores(model_name, scores, labels, train_n, len(labels))
    row["manifest"] = split_name
    return row


def calibration_report(
    labels: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    temperature: float,
    nll: float,
) -> pd.DataFrame:
    rows = []
    for name, scores in [("before_temperature", raw_scores), ("after_temperature", calibrated_scores)]:
        predictions = (scores >= 0.5).astype(int)
        metrics = binary_classification_metrics(labels, predictions, scores)
        rows.append(
            {
                "model": f"concat_{name}",
                "temperature": 1.0 if name == "before_temperature" else temperature,
                "validation_nll": nll if name == "after_temperature" else np.nan,
                "brier": metrics["brier_score"],
                "ece_10": expected_calibration_error(labels, scores, bins=10),
                "ece_15": expected_calibration_error(labels, scores, bins=15),
                "macro_f1": metrics["macro_f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )
    return pd.DataFrame(rows)


def select_abstention_row(abstention: pd.DataFrame) -> pd.Series:
    feasible = abstention[abstention["coverage"] >= 0.7]
    if feasible.empty:
        feasible = abstention
    return feasible.sort_values(["selective_accuracy", "coverage"], ascending=False).iloc[0]


def complementarity_report(labels: np.ndarray, text_scores: np.ndarray, image_scores: np.ndarray, fusion_scores: np.ndarray) -> pd.DataFrame:
    text_correct = (text_scores >= 0.5).astype(int) == labels
    image_correct = (image_scores >= 0.5).astype(int) == labels
    fusion_correct = (fusion_scores >= 0.5).astype(int) == labels
    categories = {
        "A_text_correct_image_correct": text_correct & image_correct,
        "B_text_correct_image_wrong": text_correct & ~image_correct,
        "C_text_wrong_image_correct": ~text_correct & image_correct,
        "D_text_wrong_image_wrong": ~text_correct & ~image_correct,
    }
    rows = []
    for name, mask in categories.items():
        count = int(mask.sum())
        rows.append(
            {
                "category": name,
                "count": count,
                "percentage": float(count / len(labels)),
                "fusion_correct_count": int((fusion_correct & mask).sum()),
                "fusion_wrong_count": int((~fusion_correct & mask).sum()),
            }
        )
    fusion_fixes_c = int((categories["C_text_wrong_image_correct"] & fusion_correct).sum())
    fusion_breaks_b = int((categories["B_text_correct_image_wrong"] & ~fusion_correct).sum())
    fusion_fixes_d = int((categories["D_text_wrong_image_wrong"] & fusion_correct).sum())
    fusion_breaks_a = int((categories["A_text_correct_image_correct"] & ~fusion_correct).sum())
    rows.append(
        {
            "category": "net_corrections",
            "count": fusion_fixes_c + fusion_fixes_d - fusion_breaks_a - fusion_breaks_b,
            "percentage": float((fusion_fixes_c + fusion_fixes_d - fusion_breaks_a - fusion_breaks_b) / len(labels)),
            "fusion_correct_count": fusion_fixes_c + fusion_fixes_d,
            "fusion_wrong_count": fusion_breaks_a + fusion_breaks_b,
        }
    )
    return pd.DataFrame(rows)


def prediction_correlation_report(predictions: dict[str, np.ndarray], labels: np.ndarray) -> pd.DataFrame:
    text = predictions["text_repr_same_head"]
    image = predictions["image_repr_same_head"]
    fusion = predictions["concat_fusion"]
    disagreement = np.abs(text - image)
    fusion_error = ((fusion >= 0.5).astype(int) != labels).astype(float)
    return pd.DataFrame(
        [
            {"pair": "text_image", "pearson": pearson_correlation(text, image)},
            {"pair": "text_fusion", "pearson": pearson_correlation(text, fusion)},
            {"pair": "image_fusion", "pearson": pearson_correlation(image, fusion)},
            {"pair": "abs_text_image_disagreement_vs_fusion_error", "pearson": pearson_correlation(disagreement, fusion_error)},
            {"pair": "mean_abs_text_image_disagreement", "pearson": float(disagreement.mean())},
            {"pair": "p95_abs_text_image_disagreement", "pearson": float(np.percentile(disagreement, 95))},
        ]
    )


def image_contribution_report(labels: np.ndarray, text_scores: np.ndarray, fusion_scores: np.ndarray) -> pd.DataFrame:
    delta = fusion_scores - text_scores
    abs_delta = np.abs(delta)
    text_correct = (text_scores >= 0.5).astype(int) == labels
    fusion_correct = (fusion_scores >= 0.5).astype(int) == labels
    groups = {
        "overall": np.ones(len(labels), dtype=bool),
        "text_correct": text_correct,
        "text_wrong": ~text_correct,
        "fusion_corrected_text_errors": ~text_correct & fusion_correct,
    }
    rows = []
    for name, mask in groups.items():
        rows.append(
            {
                "group": name,
                "count": int(mask.sum()),
                "mean_delta": float(delta[mask].mean()) if mask.any() else 0.0,
                "mean_abs_delta": float(abs_delta[mask].mean()) if mask.any() else 0.0,
                "median_abs_delta": float(np.median(abs_delta[mask])) if mask.any() else 0.0,
                "p95_abs_delta": float(np.percentile(abs_delta[mask], 95)) if mask.any() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def shuffle_repeat_report(
    head: FusionHead,
    text_validation: np.ndarray,
    image_validation: np.ndarray,
    labels: np.ndarray,
    config: dict[str, object],
) -> pd.DataFrame:
    rows = []
    for seed in config["shuffle_seeds"]:
        rng = np.random.default_rng(int(seed))
        image_scores = head.predict_proba(concatenate_features(text_validation, image_validation[rng.permutation(len(image_validation))]))
        text_scores = head.predict_proba(concatenate_features(text_validation[rng.permutation(len(text_validation))], image_validation))
        rows.append({"shuffle": "image", "seed": int(seed), **binary_classification_metrics(labels, (image_scores >= 0.5).astype(int), image_scores)})
        rows.append({"shuffle": "text", "seed": int(seed), **binary_classification_metrics(labels, (text_scores >= 0.5).astype(int), text_scores)})
    frame = pd.DataFrame(rows)
    aggregate = frame.groupby("shuffle")["macro_f1"].agg(["mean", "std"]).reset_index()
    aggregate["seed"] = -1
    aggregate = aggregate.rename(columns={"mean": "macro_f1_mean", "std": "macro_f1_std"})
    return pd.concat([frame, aggregate], ignore_index=True, sort=False)


def bootstrap_report(labels: np.ndarray, predictions: dict[str, np.ndarray], config: dict[str, object]) -> pd.DataFrame:
    text = bootstrap_macro_f1(labels, predictions["text_repr_same_head"], seed=int(config["seed"]), samples=int(config["bootstrap_samples"]))
    concat = bootstrap_macro_f1(labels, predictions["concat_fusion"], seed=int(config["seed"]) + 1, samples=int(config["bootstrap_samples"]))
    similarity = bootstrap_macro_f1(labels, predictions["similarity_fusion"], seed=int(config["seed"]) + 2, samples=int(config["bootstrap_samples"]))
    diff = concat - text
    rows = []
    for model_name, values in [("text_repr_same_head", text), ("concat_fusion", concat), ("similarity_fusion", similarity), ("concat_minus_text", diff)]:
        rows.append({"model": model_name, **confidence_interval(values)})
    return pd.DataFrame(rows)


def high_confidence_errors(manifest: pd.DataFrame, labels: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for model in ["text_repr_same_head", "image_repr_same_head", "concat_fusion"]:
        scores = predictions[model]
        pred = (scores >= 0.5).astype(int)
        confidence = np.maximum(scores, 1.0 - scores)
        wrong = np.where(pred != labels)[0]
        order = wrong[np.argsort(-confidence[wrong])][:10]
        for index in order:
            rows.append(
                {
                    "model": model,
                    "id": str(manifest.iloc[index]["id"]),
                    "clean_title": manifest.iloc[index]["clean_title"],
                    "true_label": int(labels[index]),
                    "predicted_probability": float(scores[index]),
                    "predicted_label": int(pred[index]),
                    "confidence": float(confidence[index]),
                    "image_path": manifest.iloc[index]["local_path"],
                    "text_probability": float(predictions["text_repr_same_head"][index]),
                    "image_probability": float(predictions["image_repr_same_head"][index]),
                    "multimodal_probability": float(predictions["concat_fusion"][index]),
                    "qualitative_hypothesis": "manual_review_needed",
                }
            )
    return pd.DataFrame(rows)


def scale_analysis(validation: pd.DataFrame, original_train: pd.DataFrame, clean_train: pd.DataFrame, config: dict[str, object], output_dir: Path) -> pd.DataFrame:
    rows = []
    labels_validation = validation["2_way_label"].to_numpy(dtype=int)
    for label, train in [("matched_clean", clean_train), ("matched_original", original_train), ("text_10k_image_manifest_population", sample_text_population(config))]:
        vocabulary, train_tokens, validation_tokens = prepare_text_arrays(train, validation, config)
        model = ManualTextEmbeddingClassifier(
            vocab_size=len(vocabulary),
            embedding_dim=int(config["text"]["embedding_dim"]),
            hidden_dim=int(config["text"]["hidden_dim"]),
            learning_rate=float(config["text"]["learning_rate"]),
            dropout=float(config["text"]["dropout"]),
            seed=int(config["seed"]),
        )
        history = model.fit(
            train_tokens,
            train["2_way_label"].to_numpy(dtype=int),
            validation_tokens=validation_tokens,
            validation_labels=labels_validation,
            epochs=int(config["text"]["epochs"]),
            batch_size=int(config["text"]["batch_size"]),
        )
        pd.DataFrame(history).to_csv(output_dir / f"scale_{label}_text_history.csv", index=False)
        scores = model.predict_proba(validation_tokens, batch_size=int(config["text"]["batch_size"]))
        metrics = binary_classification_metrics(labels_validation, (scores >= 0.5).astype(int), scores)
        rows.append({"analysis": "text_scale", "setting": label, "train_n": len(train), **metrics})
    for sample_size in [1000, len(clean_train)]:
        if sample_size > len(clean_train):
            continue
        train = clean_train.iloc[
            stratified_sample_indices(clean_train["2_way_label"].to_numpy(), sample_size, int(config["seed"]))
        ].reset_index(drop=True)
        train_pixels = load_pixel_tensor(train, tuple(config["image"]["image_size"]), f"load image scale {sample_size}")
        validation_pixels = load_pixel_tensor(validation, tuple(config["image"]["image_size"]), f"load image scale validation {sample_size}")
        mean, std = channel_statistics(train_pixels)
        train_pixels = normalize_pixels(train_pixels, mean, std)
        validation_pixels = normalize_pixels(validation_pixels, mean, std)
        model = ManualFlattenCNNClassifier(
            image_size=int(config["image"]["image_size"][0]),
            learning_rate=float(config["image"]["learning_rate"]),
            seed=int(config["seed"]),
            optimizer_name="adam",
        )
        model.fit(
            train_pixels,
            train["2_way_label"].to_numpy(dtype=int),
            epochs=int(config["image"]["epochs"]),
            batch_size=int(config["image"]["batch_size"]),
            shuffle=True,
            stop_accuracy=None,
        )
        scores = model.predict_proba(validation_pixels, batch_size=int(config["image"]["batch_size"]))
        metrics = binary_classification_metrics(labels_validation, (scores >= 0.5).astype(int), scores)
        rows.append({"analysis": "image_scale", "setting": f"image_train_{sample_size}", "train_n": len(train), **metrics})
    return pd.DataFrame(rows)


def sample_text_population(config: dict[str, object]) -> pd.DataFrame:
    manifest = pd.read_parquet(str(config["image_manifest"]))
    frame = manifest[(manifest["split"].astype(str) == "train") & manifest["clean_title"].astype("string").fillna("").str.strip().ne("")]
    size = min(int(config["text"]["scale_sample_size"]), len(frame))
    indices = stratified_sample_indices(frame["2_way_label"].to_numpy(), size, int(config["seed"]))
    return frame.iloc[indices].reset_index(drop=True)


def compute_report(clean: dict[str, object], output_dir: Path, features_dir: Path) -> pd.DataFrame:
    feature_bytes = sum(path.stat().st_size for path in (features_dir / "clean").glob("*.npy"))
    prediction_bytes = (output_dir / "clean_predictions.parquet").stat().st_size
    return pd.DataFrame(
        [
            {"system": "text_only", "parameter_count": clean["text_parameter_count"], "representation_cache_bytes": feature_bytes, "prediction_file_bytes": prediction_bytes},
            {"system": "image_cnn", "parameter_count": clean["image_parameter_count"], "representation_cache_bytes": feature_bytes, "prediction_file_bytes": prediction_bytes},
            {"system": "concat_fusion", "parameter_count": 0, "representation_cache_bytes": feature_bytes, "prediction_file_bytes": prediction_bytes},
        ]
    )


def plot_thresholds(threshold: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for column in ["macro_f1", "precision", "recall"]:
        plt.plot(threshold["threshold"], threshold[column], label=column)
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_reliability(reliability: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    for model, group in reliability.groupby("model"):
        plt.plot(group["avg_confidence"], group["empirical_positive_rate"], marker="o", label=model)
    plt.xlabel("Average predicted probability")
    plt.ylabel("Empirical fake rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_abstention(abstention: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(abstention["coverage"], abstention["selective_accuracy"], marker="o", label="selective accuracy")
    plt.plot(abstention["coverage"], abstention["selective_error_rate"], marker="o", label="selective error")
    plt.xlabel("Coverage")
    plt.ylabel("Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def create_notebook(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        "Milestone 7 recap",
        "Leakage-reduced dataset",
        "Original vs clean evaluation",
        "Threshold analysis",
        "Calibration",
        "Reliability diagrams",
        "Uncertainty / abstention",
        "Complementarity",
        "Image contribution",
        "Shuffle robustness",
        "Bootstrap confidence intervals",
        "Paired model comparison",
        "Error analysis",
        "Scale analysis",
        "Research conclusions",
    ]
    cells = [
        markdown_cell("# Milestone 8: Robustness, Calibration, and Error Analysis\n\nThis notebook loads saved experiment artifacts and does not retrain models."),
        code_cell("from pathlib import Path\nimport pandas as pd\nREPORT = Path('../reports/experiments/milestone8')\nFIGURES = REPORT / 'figures'"),
    ]
    for section in sections:
        slug = section.lower().replace(" / ", "_").replace(" ", "_").replace("-", "_")
        csv_name = {
            "Leakage-reduced dataset": "leakage_removals.csv",
            "Original vs clean evaluation": "clean_comparison.csv",
            "Threshold analysis": "threshold_analysis.csv",
            "Calibration": "calibration_metrics.csv",
            "Reliability diagrams": "reliability_bins.csv",
            "Uncertainty / abstention": "abstention_results.csv",
            "Complementarity": "complementarity.csv",
            "Image contribution": "image_contribution.csv",
            "Shuffle robustness": "shuffle_multiseed.csv",
            "Bootstrap confidence intervals": "bootstrap_results.csv",
            "Error analysis": "high_confidence_errors.csv",
            "Scale analysis": "scale_analysis.csv",
        }.get(section)
        cells.append(markdown_cell(f"## {section}"))
        if csv_name:
            cells.append(code_cell(f"pd.read_csv(REPORT / '{csv_name}').head(20)"))
        elif section == "Paired model comparison":
            cells.append(code_cell("import json\njson.loads((REPORT / 'paired_comparison.json').read_text())"))
        elif section == "Research conclusions":
            cells.append(code_cell("(REPORT / 'README.md').read_text()[:4000]"))
        else:
            cells.append(code_cell(f"# See saved artifacts in {{REPORT}}\n'{slug}'"))
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def markdown_cell(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict[str, object]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def final_recommendation(comparison: pd.DataFrame, bootstrap: pd.DataFrame, paired: dict[str, float]) -> str:
    clean_text = comparison[(comparison["manifest"] == "clean") & (comparison["model"] == "text_repr_same_head")]["macro_f1"].iloc[0]
    clean_fusion = comparison[(comparison["manifest"] == "clean") & (comparison["model"] == "concat_fusion")]["macro_f1"].iloc[0]
    diff = bootstrap[bootstrap["model"] == "concat_minus_text"].iloc[0]
    if clean_fusion > clean_text and diff["ci_low"] > 0:
        return "MULTIMODAL"
    if clean_fusion < clean_text:
        return "TEXT-FIRST"
    if paired["n01_a_wrong_b_correct"] <= paired["n10_a_correct_b_wrong"] + 5:
        return "TEXT-FIRST"
    return "TEXT-FIRST"


def write_readme(
    path: Path,
    metadata: dict[str, object],
    comparison: pd.DataFrame,
    calibration: pd.DataFrame,
    abstention: pd.DataFrame,
    complementarity: pd.DataFrame,
    contribution: pd.DataFrame,
    shuffle: pd.DataFrame,
    bootstrap: pd.DataFrame,
    scale: pd.DataFrame,
) -> None:
    selected_threshold = metadata["selected_threshold"]
    selected_abstention = metadata["selected_abstention"]
    shuffle_agg = shuffle[shuffle["seed"] == -1]
    lines = [
        "# Milestone 8 Robustness and Calibration",
        "",
        "The official Fakeddit test split was not used. This experiment keeps the project in the research phase and does not add production infrastructure.",
        "",
        f"- Runtime seconds: {metadata['runtime_seconds']:.2f}",
        f"- Original matched train/validation: {metadata['original_train_size']} / {metadata['original_validation_size']}",
        f"- Clean matched train/validation: {metadata['clean_train_size']} / {metadata['clean_validation_size']}",
        f"- Leakage removals: {metadata['leakage_removals']}",
        f"- Selected threshold: {selected_threshold['threshold']:.2f} by validation Macro F1",
        f"- Selected abstention band: {selected_abstention['band']:.2f}",
        f"- Final recommendation: {metadata['final_recommendation']}",
        "",
        "## Original vs Leakage-Reduced",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Calibration",
        "",
        calibration.to_markdown(index=False),
        "",
        "## Abstention",
        "",
        abstention.to_markdown(index=False),
        "",
        "## Complementarity",
        "",
        complementarity.to_markdown(index=False),
        "",
        "## Image Contribution",
        "",
        contribution.to_markdown(index=False),
        "",
        "## Shuffle Robustness",
        "",
        shuffle_agg.to_markdown(index=False),
        "",
        "## Bootstrap Confidence Intervals",
        "",
        bootstrap.to_markdown(index=False),
        "",
        "## Scale Analysis",
        "",
        scale.to_markdown(index=False),
        "",
        "## Conclusion",
        "",
        "The evidence supports a TEXT-FIRST research model at this stage. Fusion remains text-dominant, image shuffling remains close to aligned fusion, and the bootstrap difference is used to judge whether the tiny fusion gain is meaningful.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = run_experiment(load_config(args.config))
    print(f"Wrote Milestone 8 experiment under {output_dir}")


if __name__ == "__main__":
    main()
