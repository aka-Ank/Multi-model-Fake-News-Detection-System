"""Train and evaluate real paired CLIP fusion heads on fnews-research-v1."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fnews.ml.models.clip_fusion import ConcatFusionHead, GatedFusionHead


def runtime_device() -> torch.device:
    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "per_class": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["true", "fake"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def split_manifest(root: Path, split: str) -> pd.DataFrame:
    frame = pd.read_parquet(root / "manifests" / f"{split}_manifest.parquet")
    frame = frame.loc[
        (frame.source_dataset == "Fakeddit")
        & (frame.download_status == "verified_reused_or_processed")
    ].copy()
    if (
        frame.image_path.isna().any()
        or frame.text.isna().any()
        or frame.fake_news_label.isna().any()
    ):
        raise ValueError(f"{split} is not a complete paired Fakeddit split")
    return frame.reset_index(drop=True)


def load_embeddings(root: Path) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    frames = {split: split_manifest(root, split) for split in ("train", "validate", "test")}
    expected = {"train": 2353, "validate": 933, "test": 947}
    if {name: len(frame) for name, frame in frames.items()} != expected:
        raise ValueError("Unexpected Fakeddit research-dataset split sizes")
    archive = np.load(
        "reports/experiments/research_dataset_baselines/clip_frozen_fusion/embeddings.npz"
    )
    embeddings = {key: archive[key].astype(np.float32) for key in archive.files}
    if any(
        len(embeddings[key]) != len(frames[key]) or embeddings[key].shape[1] != 1024
        for key in frames
    ):
        raise ValueError("Cached CLIP embeddings do not align with the exact split manifests")
    return embeddings, frames


def normalize(
    train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, [(item - mean) / std for item in others], mean, std


def fit_head(
    name: str,
    head: nn.Module,
    train: np.ndarray,
    labels: np.ndarray,
    validation: np.ndarray,
    validation_labels: np.ndarray,
    config: dict[str, object],
) -> tuple[nn.Module, list[dict[str, float]], int]:
    dev = runtime_device()
    head.to(dev)
    dataset = TensorDataset(
        torch.from_numpy(train[:, :512]),
        torch.from_numpy(train[:, 512:]),
        torch.from_numpy(labels.astype(np.int64)),
    )
    data = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loss_fn = nn.CrossEntropyLoss()
    best_score, best_epoch, wait = -1.0, 0, 0
    best_state = None
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        head.train()
        losses = []
        for image, text, target in data:
            optimizer.zero_grad(set_to_none=True)
            logits = head(image.to(dev), text.to(dev))
            loss = loss_fn(logits, target.to(dev))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        predictions = predict_head(head, validation, dev)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            **{
                f"validation_{k}": float(v)
                for k, v in metrics(validation_labels, predictions).items()
                if isinstance(v, float)
            },
        }
        history.append(row)
        score = row["validation_macro_f1"]
        if score > best_score:
            best_score, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            wait += 1
            if wait >= int(config["patience"]):
                break
    assert best_state is not None
    head.load_state_dict(best_state)
    return head, history, best_epoch


def predict_head(
    head: nn.Module, features: np.ndarray, dev: torch.device | None = None
) -> np.ndarray:
    dev = dev or runtime_device()
    head.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(features), 128):
            batch = torch.from_numpy(features[start : start + 128]).to(dev)
            scores.append(head(batch[:, :512], batch[:, 512:]).argmax(1).cpu().numpy())
    return np.concatenate(scores)


def save_eval(
    output: Path, name: str, split: str, frame: pd.DataFrame, predictions: np.ndarray
) -> dict[str, object]:
    target = output / name
    target.mkdir(parents=True, exist_ok=True)
    result = metrics(frame.fake_news_label.to_numpy(int), predictions)
    pd.DataFrame(
        result["confusion_matrix"], index=["true", "fake"], columns=["pred_true", "pred_fake"]
    ).to_csv(target / f"{split}_confusion_matrix.csv")
    pd.DataFrame(
        {"sample_id": frame.sample_id, "label": frame.fake_news_label, "prediction": predictions}
    ).to_parquet(target / f"{split}_predictions.parquet", index=False)
    return result


def ablations(
    head: nn.Module, validation: np.ndarray, labels: np.ndarray, seed: int
) -> dict[str, dict[str, object]]:
    rng = np.random.default_rng(seed)
    image, text = validation[:, :512], validation[:, 512:]
    outcomes = {}
    for label, features in {
        "paired": validation,
        "shuffled_image": np.concatenate([image[rng.permutation(len(image))], text], axis=1),
        "shuffled_text": np.concatenate([image, text[rng.permutation(len(text))]], axis=1),
    }.items():
        outcomes[label] = metrics(labels, predict_head(head, features))
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/clip_fusion_research_dataset.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    raw, frames = load_embeddings(Path(config["dataset_root"]))
    train, other, mean, std = normalize(raw["train"], raw["validate"], raw["test"])
    validation, test = other
    labels = frames["train"].fake_news_label.to_numpy(int)
    validation_labels = frames["validate"].fake_news_label.to_numpy(int)
    models = {
        "concat": ConcatFusionHead(
            hidden_dim=int(config["hidden_dim"]), dropout=float(config["dropout"])
        ),
        "gated": GatedFusionHead(
            hidden_dim=int(config["hidden_dim"]), dropout=float(config["dropout"])
        ),
    }
    results = {}
    fitted = {}
    for name, head in models.items():
        head, history, epoch = fit_head(
            name, head, train, labels, validation, validation_labels, config
        )
        fitted[name] = head
        pd.DataFrame(history).to_csv(output / name / "history.csv", index=False) if (
            output / name
        ).mkdir(parents=True, exist_ok=True) is None else None
        validation_result = save_eval(
            output, name, "validation", frames["validate"], predict_head(head, validation)
        )
        results[name] = {"validation": validation_result, "best_epoch": epoch}
    selected = max(results, key=lambda item: results[item]["validation"]["macro_f1"])
    selected_head = fitted[selected]
    # The design is now frozen by validation macro-F1. Test labels are used once below.
    results[selected]["test"] = save_eval(
        output, selected, "test", frames["test"], predict_head(selected_head, test)
    )
    results[selected]["ablations"] = ablations(
        selected_head, validation, validation_labels, int(config["seed"])
    )
    torch.save(
        {
            "architecture": selected,
            "state_dict": selected_head.cpu().state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "config": config,
            "label_mapping": {0: "true", 1: "fake"},
        },
        output / selected / "best_model.pt",
    )
    metadata = {
        "dataset_version": "fnews-research-v1",
        "clip_model": config["clip_model"],
        "split_counts": {key: len(value) for key, value in frames.items()},
        "selected_model": selected,
        "results": results,
        "runtime_seconds": time.perf_counter() - started,
        "device": str(runtime_device()),
    }
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    Path(config["config_copy_path"]).write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected,
                "validation": results[selected]["validation"],
                "test": results[selected]["test"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
