"""Validation-only temperature calibration for the frozen FNEWS fusion checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedShuffleSplit

from fnews.ml.models.clip_fusion import GatedFusionHead

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports/phase2/calibration"


def ece(probability: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probability >= lower) & ((probability < upper) if upper < 1 else (probability <= upper))
        if mask.any():
            result += mask.mean() * abs(probability[mask].mean() - labels[mask].mean())
    return float(result)


def reliability(probability: np.ndarray, labels: np.ndarray) -> list[dict[str, float | int]]:
    rows = []
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:], strict=True):
        mask = (probability >= lower) & ((probability < upper) if upper < 1 else (probability <= upper))
        rows.append({"lower": lower, "upper": upper, "count": int(mask.sum()), "mean_probability": float(probability[mask].mean()) if mask.any() else float("nan"), "observed_fake_rate": float(labels[mask].mean()) if mask.any() else float("nan")})
    return rows


def scores(probability: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {"ece_10": ece(probability, labels), "brier": float(brier_score_loss(labels, probability)), "negative_log_likelihood": float(log_loss(labels, np.c_[1 - probability, probability], labels=[0, 1]))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(ROOT / "data/research_dataset/manifests/validate_manifest.parquet")
    frame = frame.loc[(frame.source_dataset == "Fakeddit") & (frame.download_status == "verified_reused_or_processed")].reset_index(drop=True)
    labels = frame.fake_news_label.to_numpy(int)
    checkpoint = torch.load(ROOT / "reports/multimodal_fusion/clip_fusion_v1/gated/best_model.pt", map_location="cpu", weights_only=False)
    raw = np.load(ROOT / "reports/experiments/research_dataset_baselines/clip_frozen_fusion/embeddings.npz")["validate"].astype(np.float32)
    feature = (raw - checkpoint["feature_mean"].astype(np.float32)) / checkpoint["feature_std"].astype(np.float32)
    head = GatedFusionHead(hidden_dim=int(checkpoint["config"]["hidden_dim"]), dropout=float(checkpoint["config"]["dropout"]))
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(feature[:, :512]), torch.from_numpy(feature[:, 512:])).numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    calibration_indices, assessment_indices = next(splitter.split(logits, labels))
    calibration_logits = torch.tensor(logits[calibration_indices], dtype=torch.float32)
    calibration_labels = torch.tensor(labels[calibration_indices], dtype=torch.long)
    log_temperature = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.CrossEntropyLoss()
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_fn(calibration_logits / log_temperature.exp(), calibration_labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    temperature = float(log_temperature.exp().detach())
    raw_probability = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    calibrated_probability = torch.softmax(torch.tensor(logits / temperature), dim=1)[:, 1].numpy()
    assessment = assessment_indices
    output = {"protocol": {"split": "Fakeddit validation only", "calibration_samples": len(calibration_indices), "assessment_samples": len(assessment_indices), "seed": 42, "heldout_test_accessed": False}, "temperature": temperature, "assessment_raw": scores(raw_probability[assessment], labels[assessment]), "assessment_calibrated": scores(calibrated_probability[assessment], labels[assessment]), "calibration_raw": scores(raw_probability[calibration_indices], labels[calibration_indices]), "calibration_calibrated": scores(calibrated_probability[calibration_indices], labels[calibration_indices])}
    pd.DataFrame(reliability(raw_probability[assessment], labels[assessment])).to_csv(OUT / "assessment_raw_reliability.csv", index=False)
    pd.DataFrame(reliability(calibrated_probability[assessment], labels[assessment])).to_csv(OUT / "assessment_calibrated_reliability.csv", index=False)
    pd.DataFrame({"sample_id": frame.sample_id, "label": labels, "partition": np.where(np.isin(np.arange(len(frame)), calibration_indices), "temperature_fit", "assessment"), "raw_fake_probability": raw_probability, "calibrated_fake_probability": calibrated_probability}).to_parquet(OUT / "validation_probabilities.parquet", index=False)
    (OUT / "temperature.json").write_text(json.dumps({"temperature": temperature, "checkpoint": "fakeddit-clip-gated-fusion-v1", "fit_partition": "validation_seed42_stratified_half"}, indent=2), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
