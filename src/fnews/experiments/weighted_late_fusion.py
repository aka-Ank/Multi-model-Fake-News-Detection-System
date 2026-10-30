"""Validation-only weighted late fusion of saved SVM and ResNet-18 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from torchvision.models import ResNet18_Weights, resnet18

from fnews.experiments.research_dataset_baselines import loader

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/research_dataset"
BASE = ROOT / "reports/experiments/research_dataset_baselines"
OUT = ROOT / "reports/weighted_late_fusion_v1"


def load(split: str) -> pd.DataFrame:
    frame = pd.read_parquet(DATA / "manifests" / f"{split}_manifest.parquet")
    frame = frame.loc[
        frame.source_dataset.eq("Fakeddit")
        & frame.task_type.eq("fake_news_classification")
        & frame.download_status.eq("verified_reused_or_processed")
    ].copy()
    frame["label"] = frame.fake_news_label.astype(int)
    expected = {"train": 2353, "validate": 933}[split]
    if len(frame) != expected or frame.text.isna().any() or frame.image_path.isna().any():
        raise ValueError(f"Unexpected paired {split} manifest")
    return frame.reset_index(drop=True)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(0.1, 1, bins), strict=True):
        mask = (p >= lower) & ((p < upper) if upper < 1 else (p <= upper))
        if mask.any():
            result += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(result)


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.c_[1 - p, p], labels=[0, 1])),
        "ece_10": ece(y, p),
    }


def predictions(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        **probability_metrics(y, p),
    }


def resnet_probabilities(frame: pd.DataFrame) -> np.ndarray:
    checkpoint = torch.load(BASE / "resnet18_frozen/best_model.pt", map_location="cpu", weights_only=True)
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    values = []
    with torch.no_grad():
        for pixels, _, _ in loader(frame, DATA, ResNet18_Weights.DEFAULT.transforms(), 32):
            logits = model(pixels.to(device))
            values.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(values)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train, validation = load("train"), load("validate")
    y_train, y_val = train.label.to_numpy(), validation.label.to_numpy()
    base = joblib.load(BASE / "tfidf_svm/model.joblib")
    oof = np.empty(len(train), dtype=float)
    for fit, holdout in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train.text, y_train):
        model = clone(base).fit(train.text.iloc[fit], y_train[fit])
        oof[holdout] = model.decision_function(train.text.iloc[holdout])
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=42).fit(oof.reshape(-1, 1), y_train)
    text_margin = base.decision_function(validation.text)
    text_probability = calibrator.predict_proba(text_margin.reshape(-1, 1))[:, 1]
    image_probability = resnet_probabilities(validation)
    if validation.sample_id.nunique() != len(validation):
        raise ValueError("Paired ID alignment failed")
    rows = []
    for alpha in np.linspace(0, 1, 101):
        probability = alpha * text_probability + (1 - alpha) * image_probability
        rows.append({"alpha": float(alpha), **predictions(y_val, probability)})
    sweep = pd.DataFrame(rows)
    best = sweep.sort_values(["macro_f1", "accuracy", "alpha"], ascending=[False, False, False]).iloc[0]
    selected_alpha = float(best.alpha)
    fused = selected_alpha * text_probability + (1 - selected_alpha) * image_probability
    output = validation[["sample_id", "label"]].copy()
    output["text_margin"] = text_margin
    output["text_probability_fake"] = text_probability
    output["image_probability_fake"] = image_probability
    output["combined_probability_fake"] = fused
    output["prediction"] = (fused >= 0.5).astype(int)
    output.to_parquet(OUT / "validation_predictions.parquet", index=False)
    sweep.to_csv(OUT / "alpha_sweep.csv", index=False)
    joblib.dump(calibrator, OUT / "svm_platt_calibrator.joblib")
    payload = {
        "protocol": "5-fold out-of-fold train-margin Platt calibration; validation used once for alpha selection",
        "split": "validation-933", "paired_ids": int(len(output)), "alpha": selected_alpha,
        "text_calibration_oof_train": probability_metrics(y_train, calibrator.predict_proba(oof.reshape(-1, 1))[:, 1]),
        "validation": {
            "image": predictions(y_val, image_probability),
            "text": predictions(y_val, text_probability),
            "alpha_0_5": predictions(y_val, 0.5 * text_probability + 0.5 * image_probability),
            "selected": predictions(y_val, fused),
        },
        "gated_fusion_matched_validation": {"accuracy": 0.8745980707395499, "macro_f1": 0.8706998560844305},
        "limitations": "No held-out test data accessed. Image softmax probabilities were assessed on validation but not calibrated or tuned.",
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
