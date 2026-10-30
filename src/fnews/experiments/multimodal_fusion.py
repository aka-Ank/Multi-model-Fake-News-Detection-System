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
from fnews.experiments.neural_text import (
    build_text_vocabulary,
    encode_documents,
    sequence_length_stats,
)
from fnews.ml.metrics.classification import binary_classification_metrics, confusion_matrix_binary
from fnews.ml.models.cnn import ManualFlattenCNNClassifier
from fnews.ml.models.multimodal import (
    FusionHead,
    concatenate_features,
    gated_features,
    similarity_features,
    validate_alignment,
)
from fnews.ml.models.text_embedding import ManualTextEmbeddingClassifier
from fnews.ml.preprocessing.text import tokenize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched-sample frozen multimodal fusion experiments.")
    parser.add_argument("--config", default="configs/experiments/multimodal_fusion.json")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_experiment(config: dict[str, object]) -> Path:
    started = time.perf_counter()
    output_dir = Path(str(config["output_dir"]))
    features_dir = Path(str(config["features_dir"]))
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    train_manifest, validation_manifest = build_matched_manifests(config)
    train_manifest.to_parquet(config["train_manifest_output"], index=False)
    validation_manifest.to_parquet(config["validation_manifest_output"], index=False)
    audit = dataset_audit(train_manifest, validation_manifest, config)
    audit.to_csv(output_dir / "dataset_audit.csv", index=False)

    vocabulary, train_tokens, validation_tokens = prepare_text_arrays(train_manifest, validation_manifest, config)
    train_labels = train_manifest["2_way_label"].to_numpy(dtype=int)
    validation_labels = validation_manifest["2_way_label"].to_numpy(dtype=int)
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
        train_labels,
        validation_tokens=validation_tokens,
        validation_labels=validation_labels,
        epochs=int(config["text"]["epochs"]),
        batch_size=int(config["text"]["batch_size"]),
    )
    pd.DataFrame(text_history).to_csv(output_dir / "text_history.csv", index=False)

    train_pixels = load_pixel_tensor(train_manifest, tuple(config["image"]["image_size"]), "load matched train images")
    validation_pixels = load_pixel_tensor(
        validation_manifest,
        tuple(config["image"]["image_size"]),
        "load matched validation images",
    )
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
        train_labels,
        epochs=int(config["image"]["epochs"]),
        batch_size=int(config["image"]["batch_size"]),
        shuffle=True,
        stop_accuracy=None,
    )
    pd.DataFrame(image_history).to_csv(output_dir / "image_history.csv", index=False)

    text_train = text_model.encode_text(train_tokens, training=False)
    text_validation = text_model.encode_text(validation_tokens, training=False)
    image_train = image_model.encode_image(train_pixels, training=False)
    image_validation = image_model.encode_image(validation_pixels, training=False)
    save_feature_cache(features_dir, train_manifest, validation_manifest, text_train, text_validation, image_train, image_validation)
    validate_alignment(
        np.load(features_dir / "ids_train.npy", allow_pickle=True),
        train_manifest["id"].astype(str).to_numpy(),
        np.load(features_dir / "labels_train.npy"),
        train_labels,
    )

    rows = []
    rows.append(evaluate_scores("text_encoder_classifier", text_model.predict_proba(validation_tokens), validation_labels, len(train_labels), len(validation_labels)))
    rows.append(evaluate_scores("image_cnn_classifier", image_model.predict_proba(validation_pixels), validation_labels, len(train_labels), len(validation_labels)))

    fusion_sets = {
        "text_repr_same_head": (text_train, text_validation),
        "image_repr_same_head": (image_train, image_validation),
        "concat_fusion": (concatenate_features(text_train, image_train), concatenate_features(text_validation, image_validation)),
        "similarity_fusion": (similarity_features(text_train, image_train), similarity_features(text_validation, image_validation)),
    }
    gated_train, gate_train = gated_features(text_train, image_train)
    gated_validation, gate_validation = gated_features(text_validation, image_validation)
    fusion_sets["gated_fusion"] = (gated_train, gated_validation)

    histories = {}
    predictions = {}
    heads = {}
    for name, (train_x, val_x) in fusion_sets.items():
        head = FusionHead(
            input_dim=train_x.shape[1],
            hidden_dim=int(config["fusion"]["hidden_dim"]),
            learning_rate=float(config["fusion"]["learning_rate"]),
            dropout=float(config["fusion"]["dropout"]),
            seed=int(config["seed"]),
        )
        history = head.fit(
            train_x,
            train_labels,
            validation_features=val_x,
            validation_labels=validation_labels,
            epochs=int(config["fusion"]["epochs"]),
            batch_size=int(config["fusion"]["batch_size"]),
        )
        histories[name] = pd.DataFrame(history)
        heads[name] = head
        histories[name].to_csv(output_dir / f"{name}_history.csv", index=False)
        scores = head.predict_proba(val_x)
        predictions[name] = scores
        rows.append(evaluate_scores(name, scores, validation_labels, len(train_labels), len(validation_labels)))

    rows.extend(
        modality_ablations(
            heads["concat_fusion"],
            text_validation,
            image_validation,
            validation_labels,
            int(config["seed"]),
        )
    )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    pd.DataFrame(confusion_matrix_binary(validation_labels, (predictions["concat_fusion"] >= 0.5).astype(int))).to_csv(
        output_dir / "concat_confusion_matrix.csv",
        index=False,
    )
    gate_report = gate_analysis(gate_validation, validation_labels, predictions["gated_fusion"])
    gate_report.to_csv(output_dir / "gate_analysis.csv", index=False)
    plot_gate_histogram(gate_validation, figures_dir / "gate_histogram.png")

    metadata = {
        "config": config,
        "runtime_seconds": time.perf_counter() - started,
        "train_size": len(train_manifest),
        "validation_size": len(validation_manifest),
        "text_dim": int(text_train.shape[1]),
        "image_dim": int(image_train.shape[1]),
        "channel_mean": channel_mean.tolist(),
        "channel_std": channel_std.tolist(),
        "best_macro_f1": float(comparison["macro_f1"].max()),
        "comparison": comparison.to_dict(orient="records"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    Path(str(config["config_copy_path"])).write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    write_readme(output_dir / "README.md", metadata, comparison, audit, gate_report)
    return output_dir


def build_matched_manifests(config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_parquet(str(config["image_manifest"]))
    valid = manifest[
        manifest["valid_image"]
        & manifest["local_path"].notna()
        & manifest["clean_title"].astype("string").fillna("").str.strip().ne("")
    ].copy()
    columns = [
        "id",
        "split",
        "clean_title",
        "2_way_label",
        "image_url",
        "local_path",
        "domain",
        "subreddit",
        "sha256",
        "perceptual_hash",
    ]
    for column in ["domain", "subreddit"]:
        if column not in valid.columns:
            valid[column] = None
    keep = [column for column in columns if column in valid.columns]
    train = valid[valid["split"].astype(str) == "train"][keep].sort_values("id").reset_index(drop=True)
    validation = valid[valid["split"].astype(str) == "validate"][keep].sort_values("id").reset_index(drop=True)
    return train, validation


def dataset_audit(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    full = pd.read_parquet(str(config["image_manifest"]))
    rows = []
    for name, frame in [
        ("matched_train", train),
        ("matched_validation", validation),
        ("full_multimodal_train", full[full["split"].astype(str) == "train"]),
        ("full_multimodal_validation", full[full["split"].astype(str) == "validate"]),
    ]:
        counts = frame["2_way_label"].value_counts().sort_index()
        rows.append(
            {
                "group": name,
                "rows": len(frame),
                "true_0": int(counts.get(0, 0)),
                "fake_1": int(counts.get(1, 0)),
                "fake_pct": float(counts.get(1, 0) / len(frame)) if len(frame) else 0.0,
                "duplicate_titles": int(frame["clean_title"].astype(str).duplicated().sum()) if "clean_title" in frame else 0,
                "duplicate_image_urls": int(frame["image_url"].astype(str).duplicated().sum()) if "image_url" in frame else 0,
            }
        )
    rows.append({"group": "cross_split_sha256_overlap", "rows": len(set(train["sha256"].dropna()) & set(validation["sha256"].dropna()))})
    rows.append(
        {
            "group": "cross_split_phash_overlap",
            "rows": len(set(train["perceptual_hash"].dropna()) & set(validation["perceptual_hash"].dropna())),
        }
    )
    return pd.DataFrame(rows)


def prepare_text_arrays(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, object]):
    train_tokens_raw = [tokenize(text, mode=str(config["text"]["tokenization_mode"])) for text in train["clean_title"]]
    validation_tokens_raw = [tokenize(text, mode=str(config["text"]["tokenization_mode"])) for text in validation["clean_title"]]
    stats = sequence_length_stats(train_tokens_raw)
    max_length = int(config["text"].get("max_length") or stats["p95"])
    vocabulary = build_text_vocabulary(
        train_tokens_raw,
        min_frequency=int(config["text"]["min_frequency"]),
        max_vocabulary_size=int(config["text"]["max_vocabulary_size"]),
    )
    return vocabulary, encode_documents(train_tokens_raw, vocabulary, max_length), encode_documents(validation_tokens_raw, vocabulary, max_length)


def save_feature_cache(
    features_dir: Path,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    text_train: np.ndarray,
    text_validation: np.ndarray,
    image_train: np.ndarray,
    image_validation: np.ndarray,
) -> None:
    np.save(features_dir / "text_train_embeddings.npy", text_train)
    np.save(features_dir / "text_validate_embeddings.npy", text_validation)
    np.save(features_dir / "image_train_embeddings.npy", image_train)
    np.save(features_dir / "image_validate_embeddings.npy", image_validation)
    np.save(features_dir / "ids_train.npy", train["id"].astype(str).to_numpy())
    np.save(features_dir / "ids_validate.npy", validation["id"].astype(str).to_numpy())
    np.save(features_dir / "labels_train.npy", train["2_way_label"].to_numpy(dtype=int))
    np.save(features_dir / "labels_validate.npy", validation["2_way_label"].to_numpy(dtype=int))


def evaluate_scores(name: str, scores: np.ndarray, labels: np.ndarray, train_n: int, validation_n: int) -> dict[str, float | str]:
    predictions = (scores >= 0.5).astype(int)
    metrics = binary_classification_metrics(labels, predictions, scores)
    return {
        "model": name,
        "train_n": train_n,
        "validation_n": validation_n,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "roc_auc": metrics["roc_auc"],
        "brier_score": metrics["brier_score"],
    }


def modality_ablations(
    head: FusionHead,
    text_validation: np.ndarray,
    image_validation: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(seed)
    text_shuffled = text_validation[rng.permutation(len(text_validation))]
    image_shuffled = image_validation[rng.permutation(len(image_validation))]
    zero_text = np.zeros_like(text_validation)
    zero_image = np.zeros_like(image_validation)
    noise_image = rng.normal(0, 1, size=image_validation.shape)
    variants = {
        "concat_eval_image_shuffled": concatenate_features(text_validation, image_shuffled),
        "concat_eval_text_shuffled": concatenate_features(text_shuffled, image_validation),
        "concat_eval_zero_image": concatenate_features(text_validation, zero_image),
        "concat_eval_zero_text": concatenate_features(zero_text, image_validation),
        "concat_eval_noise_image": concatenate_features(text_validation, noise_image),
    }
    return [
        evaluate_scores(name, head.predict_proba(features), labels, len(labels), len(labels))
        for name, features in variants.items()
    ]


def gate_analysis(gate: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    predictions = (scores >= 0.5).astype(int)
    mean_gate = gate.mean(axis=1)
    groups = {
        "overall": np.ones(len(labels), dtype=bool),
        "true_label": labels == 0,
        "fake_label": labels == 1,
        "correct": predictions == labels,
        "incorrect": predictions != labels,
    }
    return pd.DataFrame(
        [
            {
                "group": name,
                "count": int(mask.sum()),
                "mean_gate": float(mean_gate[mask].mean()) if mask.any() else 0.0,
                "median_gate": float(np.median(mean_gate[mask])) if mask.any() else 0.0,
                "std_gate": float(mean_gate[mask].std()) if mask.any() else 0.0,
            }
            for name, mask in groups.items()
        ]
    )


def plot_gate_histogram(gate: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.hist(gate.mean(axis=1), bins=30)
    plt.xlabel("Mean gate value")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_readme(path: Path, metadata: dict[str, object], comparison: pd.DataFrame, audit: pd.DataFrame, gate: pd.DataFrame) -> None:
    lines = [
        "# Matched Multimodal Fusion",
        "",
        "All text-only, image-only, and fusion rows use the same matched local-image Fakeddit IDs. The official test split is not used.",
        "",
        f"- Runtime seconds: {metadata['runtime_seconds']:.2f}",
        f"- Train matched samples: {metadata['train_size']}",
        f"- Validation matched samples: {metadata['validation_size']}",
        f"- Text representation dim: {metadata['text_dim']}",
        f"- Image representation dim: {metadata['image_dim']}",
        "",
        "## Dataset Audit",
        "",
        audit.to_markdown(index=False),
        "",
        "## Model Comparison",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Gate Analysis",
        "",
        gate.to_markdown(index=False),
        "",
        "These are frozen-representation fusion experiments. Improvements or failures should be interpreted as evidence about whether these learned-from-scratch representations contain complementary signal, not as proof of factual verification.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = run_experiment(load_config(args.config))
    print(f"Wrote multimodal fusion experiment under {output_dir}")


if __name__ == "__main__":
    main()
