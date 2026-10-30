from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2

from fnews.experiments.image_research_audit import hardware_snapshot
from fnews.ml.metrics.classification import binary_classification_metrics, confusion_matrix_binary
from fnews.ml.preprocessing.split import stratified_split_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a leakage-aware frozen ResNet-18 image baseline.")
    parser.add_argument("--config", default="configs/experiments/image_transfer_resnet18_frozen.json")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_development_split(frame: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_indices, development_indices = stratified_split_indices(
        frame["2_way_label"].to_numpy(dtype=int), (1.0 - fraction, fraction), seed
    )
    return (
        frame.iloc[train_indices].reset_index(drop=True),
        frame.iloc[development_indices].reset_index(drop=True),
    )


class ManifestImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, frame: pd.DataFrame, transform: v2.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        with Image.open(row["local_path"]) as image:
            rgb = image.convert("RGB")
            tensor = self.transform(rgb)
        return tensor, torch.tensor(int(row["2_way_label"]), dtype=torch.long)


@dataclass
class RunResult:
    metrics: dict[str, float]
    scores: np.ndarray
    predictions: np.ndarray


def make_transforms(image_size: int, training: bool = False) -> v2.Compose:
    operations: list[nn.Module] = []
    if training:
        operations.extend(
            [
                v2.RandomResizedCrop(image_size, scale=(0.72, 1.0), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            ]
        )
    else:
        operations.append(v2.Resize((image_size, image_size), antialias=True))
    operations.extend(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return v2.Compose(operations)


def make_loader(
    frame: pd.DataFrame, transform: v2.Compose, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ManifestImageDataset(frame, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def build_model(pretrained: bool, trainable_blocks: str) -> tuple[nn.Module, str]:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if trainable_blocks == "layer4":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    elif trainable_blocks == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif trainable_blocks != "head":
        raise ValueError("trainable_blocks must be 'head', 'layer4', or 'all'")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    label = "ImageNet-1K ResNet-18" if pretrained else "random ResNet-18"
    return model, f"{label}; trainable={trainable_blocks}"


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> RunResult:
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for images, batch_labels in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            scores.append(probabilities.detach().cpu().numpy())
            labels.append(batch_labels.numpy())
    y_score = np.concatenate(scores)
    y_true = np.concatenate(labels).astype(int)
    y_pred = (y_score >= 0.5).astype(int)
    return RunResult(binary_classification_metrics(y_true, y_pred, y_score), y_score, y_pred)


def train_probe(
    model: nn.Module,
    train_loader: DataLoader,
    development_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    class_weights: torch.Tensor,
    patience: int,
    minimum_learning_rate: float,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor], int]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=minimum_learning_rate
    )
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_macro_f1 = -1.0
    best_epoch = 0
    patience_used = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        development = evaluate(model, development_loader, device)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "development_accuracy": development.metrics["accuracy"],
            "development_macro_f1": development.metrics["macro_f1"],
            "development_roc_auc": development.metrics["roc_auc"],
        }
        history.append(row)
        scheduler.step()
        if development.metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = development.metrics["macro_f1"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch
            patience_used = 0
        else:
            patience_used += 1
        if patience_used >= patience:
            break
    return history, best_state, best_epoch


def write_predictions(path: Path, frame: pd.DataFrame, result: RunResult) -> None:
    predictions = frame[["id", "2_way_label", "local_path"]].copy()
    predictions["score_fake"] = result.scores
    predictions["prediction"] = result.predictions
    predictions.to_parquet(path, index=False)


def plot_history(history: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(history["epoch"], history["train_loss"], label="train loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Train loss")
    right = axis.twinx()
    right.plot(history["epoch"], history["development_macro_f1"], color="tab:orange", label="development Macro F1")
    right.set_ylabel("Development Macro F1")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_readme(path: Path, metadata: dict[str, object], audit_metrics: dict[str, float]) -> None:
    lines = [
        "# ResNet-18 Image Transfer Baseline",
        "",
        "This is a pixels-only transfer-learning baseline. It excludes dimensions, aspect ratio, URL, source, "
        "subreddit, file size, and text/title fields from model inputs.",
        "Training uses random resized crops, horizontal flips, and modest color jitter; evaluation uses deterministic resizing.",
        "",
        f"- Encoder: {metadata['encoder']}",
        f"- Device: {metadata['device']}",
        f"- Training images: {metadata['train_size']}",
        f"- Development images: {metadata['development_size']}",
        f"- Audit validation images: {metadata['audit_size']}",
        f"- Best development epoch: {metadata['best_epoch']}",
        f"- Runtime seconds: {metadata['runtime_seconds']:.2f}",
        "",
        "## Audit Validation Metrics",
        "",
        pd.DataFrame([audit_metrics]).to_markdown(index=False),
        "",
        "## Protocol Note",
        "",
        "The clean matched training manifest is split internally for training and development selection. "
        "The original Fakeddit validation manifest is used as a post-selection audit only; the official test split remains sealed.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(config: dict[str, object]) -> Path:
    started = time.perf_counter()
    set_seed(int(config["seed"]))
    output = Path(str(config["output_dir"]))
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    train_full = pd.read_parquet(str(config["train_manifest"]))
    audit_frame = pd.read_parquet(str(config["audit_manifest"]))
    train_frame, development_frame = create_development_split(
        train_full, float(config["development_fraction"]), int(config["seed"])
    )
    image_size = int(config["image_size"])
    seed = int(config["seed"])
    train_loader = make_loader(
        train_frame, make_transforms(image_size, training=True), int(config["batch_size"]), True, seed
    )
    evaluation_transform = make_transforms(image_size)
    development_loader = make_loader(
        development_frame, evaluation_transform, int(config["batch_size"]), False, seed
    )
    audit_loader = make_loader(audit_frame, evaluation_transform, int(config["batch_size"]), False, seed)
    labels = train_frame["2_way_label"].to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=2)
    class_weights = torch.tensor(len(labels) / (2 * counts), dtype=torch.float32)
    model, encoder = build_model(bool(config["pretrained"]), str(config.get("trainable_blocks", "head")))
    device = select_device()
    history, best_state, best_epoch = train_probe(
        model,
        train_loader,
        development_loader,
        device,
        int(config["epochs"]),
        float(config["learning_rate"]),
        float(config["weight_decay"]),
        class_weights,
        int(config["early_stopping_patience"]),
        float(config.get("minimum_learning_rate", 0.0)),
    )
    model.load_state_dict(best_state)
    model.to(device)
    audit = evaluate(model, audit_loader, device)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output / "history.csv", index=False)
    write_predictions(output / "audit_predictions.parquet", audit_frame, audit)
    pd.DataFrame(confusion_matrix_binary(audit_frame["2_way_label"].to_numpy(), audit.predictions)).to_csv(
        output / "audit_confusion_matrix.csv", index=False
    )
    plot_history(history_frame, figures / "training_curve.png")
    checkpoint = output / "best_model.pt"
    torch.save({"state_dict": best_state, "config": config, "class_weights": class_weights.tolist()}, checkpoint)
    metadata = {
        "config": config,
        "encoder": encoder,
        "device": str(device),
        "train_size": len(train_frame),
        "development_size": len(development_frame),
        "audit_size": len(audit_frame),
        "best_epoch": best_epoch,
        "class_weights": class_weights.tolist(),
        "runtime_seconds": time.perf_counter() - started,
        "hardware": hardware_snapshot(),
        "audit_validation_metrics": audit.metrics,
        "checkpoint": str(checkpoint),
    }
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    write_readme(output / "README.md", metadata, audit.metrics)
    return output


def main() -> None:
    args = parse_args()
    output = run_experiment(load_config(args.config))
    print(f"Wrote transfer baseline under {output}")


if __name__ == "__main__":
    main()
