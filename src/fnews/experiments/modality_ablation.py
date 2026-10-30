"""Frozen-checkpoint modality sensitivity analysis on the Fakeddit validation split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from fnews.ml.models.clip_fusion import GatedFusionHead

ROOT = Path(__file__).resolve().parents[3]
EMBEDDINGS = ROOT / "reports/experiments/research_dataset_baselines/clip_frozen_fusion/embeddings.npz"
CHECKPOINT = ROOT / "reports/multimodal_fusion/clip_fusion_v1/gated/best_model.pt"
MANIFEST = ROOT / "data/research_dataset/manifests/validate_manifest.parquet"
OUTPUT = ROOT / "reports/phase2/modality_ablation"


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
    }


def forward(head: GatedFusionHead, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        image, text = torch.from_numpy(features[:, :512]), torch.from_numpy(features[:, 512:])
        image_projected = head.image_projection(image)
        text_projected = head.text_projection(text)
        gate = head.gate(torch.cat((image_projected, text_projected), dim=1))
        logits = head.classifier(gate * text_projected + (1.0 - gate) * image_projected)
        probability = torch.softmax(logits, dim=1)[:, 1]
    return (
        probability.numpy(),
        logits.argmax(1).numpy(),
        gate.numpy(),
        np.stack((image_projected.norm(dim=1).numpy(), text_projected.norm(dim=1).numpy()), axis=1),
    )


def bootstrap_delta(y: np.ndarray, left: np.ndarray, right: np.ndarray, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(500):
        indices = rng.integers(0, len(y), len(y))
        values.append(
            f1_score(y[indices], left[indices], average="macro", zero_division=0)
            - f1_score(y[indices], right[indices], average="macro", zero_division=0)
        )
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(MANIFEST)
    frame = frame.loc[(frame.source_dataset == "Fakeddit") & (frame.download_status == "verified_reused_or_processed")].reset_index(drop=True)
    archive = np.load(EMBEDDINGS)
    raw = archive["validate"].astype(np.float32)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    head = GatedFusionHead(hidden_dim=int(checkpoint["config"]["hidden_dim"]), dropout=float(checkpoint["config"]["dropout"]))
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    mean, std = checkpoint["feature_mean"].astype(np.float32), checkpoint["feature_std"].astype(np.float32)
    normalized = (raw - mean) / std
    labels = frame.fake_news_label.to_numpy(int)
    rng = np.random.default_rng(42)
    conditions = {
        "paired": normalized,
        "shuffled_image": np.concatenate([normalized[rng.permutation(len(frame)), :512], normalized[:, 512:]], axis=1),
        "shuffled_text": np.concatenate([normalized[:, :512], normalized[rng.permutation(len(frame)), 512:]], axis=1),
    }
    output: dict[str, dict[str, object]] = {}
    paired_probability = paired_prediction = None
    for name, features in conditions.items():
        probability, prediction, gate, norms = forward(head, features)
        if name == "paired":
            paired_probability, paired_prediction = probability, prediction
        assert paired_probability is not None and paired_prediction is not None
        output[name] = {
            **metric(labels, prediction),
            "prediction_change_rate_vs_paired": float(np.mean(prediction != paired_prediction)),
            "mean_absolute_probability_shift_vs_paired": float(np.mean(np.abs(probability - paired_probability))),
            "gate_mean_text_weight": float(gate.mean()),
            "gate_std_text_weight": float(gate.std()),
            "image_projection_norm_mean": float(norms[:, 0].mean()),
            "text_projection_norm_mean": float(norms[:, 1].mean()),
            "class_prediction_change_rate": {str(label): float(np.mean(prediction[labels == label] != paired_prediction[labels == label])) for label in (0, 1)},
        }
        pd.DataFrame({"sample_id": frame.sample_id, "label": labels, "prediction": prediction, "fake_probability": probability, "mean_gate_text_weight": gate.mean(axis=1)}).to_parquet(OUTPUT / f"{name}_predictions.parquet", index=False)
    shuffled_image_predictions = pd.read_parquet(
        OUTPUT / "shuffled_image_predictions.parquet"
    ).prediction.to_numpy()
    shuffled_text_predictions = pd.read_parquet(
        OUTPUT / "shuffled_text_predictions.parquet"
    ).prediction.to_numpy()
    output["paired"]["bootstrap_macro_f1_delta_vs_shuffled_image_95ci"] = bootstrap_delta(
        labels, paired_prediction, shuffled_image_predictions
    )
    output["paired"]["bootstrap_macro_f1_delta_vs_shuffled_text_95ci"] = bootstrap_delta(
        labels, paired_prediction, shuffled_text_predictions
    )
    baselines = {}
    for name in ("pretrained_text_frozen", "resnet18_frozen"):
        path = ROOT / f"reports/experiments/research_dataset_baselines/{name}/validation_predictions.parquet"
        if path.is_file():
            predicted = pd.read_parquet(path)
            if list(predicted.sample_id) == list(frame.sample_id):
                baselines[name] = metric(labels, predicted.prediction.to_numpy(int))
    result = {"split": "Fakeddit validation", "samples": len(frame), "seed": 42, "conditions": output, "matched_unimodal_baselines": baselines, "limitations": "Shuffles are controlled sensitivity interventions, not naturally occurring misinformation or verified image-caption mismatch labels."}
    (OUTPUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
