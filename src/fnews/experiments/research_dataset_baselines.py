"""Baselines evaluated only on the bounded research-dataset split manifests."""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_splits(root: Path) -> dict[str, pd.DataFrame]:
    splits = {
        name: pd.read_parquet(root / "manifests" / f"{name}_manifest.parquet")
        for name in ("train", "validate", "test")
    }
    result: dict[str, pd.DataFrame] = {}
    for name, frame in splits.items():
        current = frame.loc[
            frame["source_dataset"].eq("Fakeddit")
            & frame["task_type"].eq("fake_news_classification")
            & frame["download_status"].eq("verified_reused_or_processed")
        ].copy()
        current["label"] = current["fake_news_label"].astype(int)
        if current["text"].isna().any() or current["image_path"].isna().any():
            raise ValueError(f"{name} contains a missing modality")
        result[name] = current.reset_index(drop=True)
    expected = {"train": 2353, "validate": 933, "test": 947}
    actual = {name: len(frame) for name, frame in result.items()}
    if actual != expected:
        raise ValueError(f"Unexpected research-dataset split counts: {actual}; expected {expected}")
    return result


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["true", "fake"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def persist_evaluation(
    output: Path, name: str, split: str, frame: pd.DataFrame, predictions: np.ndarray
) -> dict[str, Any]:
    result = metric_bundle(frame.label.to_numpy(), predictions)
    target = output / name
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        result["confusion_matrix"], index=["true", "fake"], columns=["pred_true", "pred_fake"]
    ).to_csv(target / f"{split}_confusion_matrix.csv")
    pd.DataFrame(
        {"sample_id": frame.sample_id, "label": frame.label, "prediction": predictions}
    ).to_parquet(target / f"{split}_predictions.parquet", index=False)
    return result


def text_svm(splits: dict[str, pd.DataFrame], output: Path) -> dict[str, Any]:
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True
                ),
            ),
            ("svm", LinearSVC(C=1.0)),
        ]
    )
    pipeline.fit(splits["train"].text, splits["train"].label)
    joblib.dump(pipeline, output / "tfidf_svm" / "model.joblib") if (output / "tfidf_svm").mkdir(
        parents=True, exist_ok=True
    ) is None else None
    return {
        "validation": persist_evaluation(
            output,
            "tfidf_svm",
            "validation",
            splits["validate"],
            pipeline.predict(splits["validate"].text),
        )
    }


class ImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, root: Path, transform) -> None:
        self.frame, self.root, self.transform = frame.reset_index(drop=True), root, transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(self.root / row.image_path) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, torch.tensor(int(row.label)), index


def loader(
    frame: pd.DataFrame, root: Path, transform, batch_size: int, shuffle: bool = False
) -> DataLoader:
    return DataLoader(
        ImageDataset(frame, root, transform), batch_size=batch_size, shuffle=shuffle, num_workers=0
    )


def amp_context(runtime: torch.device):
    return (
        torch.autocast(device_type=runtime.type, dtype=torch.float16)
        if runtime.type in {"cuda", "mps"}
        else nullcontext()
    )


def train_head(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    runtime: torch.device,
    epochs: int,
    patience: int,
):
    model.to(runtime)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    best, best_epoch, wait = -1.0, 0, 0
    state: dict[str, torch.Tensor] | None = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for pixels, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with amp_context(runtime):
                loss = criterion(model(pixels.to(runtime)), labels.to(runtime))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        y_true, y_pred = predict_model(model, validation_loader, runtime)
        score = f1_score(y_true, y_pred, average="macro")
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_macro_f1": float(score),
            }
        )
        if score > best:
            best, best_epoch, wait = score, epoch, 0
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    assert state is not None
    model.load_state_dict(state)
    return history, best_epoch


def predict_model(
    model: nn.Module, data_loader: DataLoader, runtime: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, predictions = [], []
    with torch.no_grad():
        for pixels, batch_labels, _ in data_loader:
            with amp_context(runtime):
                logits = model(pixels.to(runtime))
            labels.extend(batch_labels.numpy().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    return np.asarray(labels), np.asarray(predictions)


def resnet_baseline(
    splits: dict[str, pd.DataFrame], root: Path, output: Path, config: dict[str, Any]
) -> dict[str, Any]:
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    model = resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    runtime = device()
    train_loader = loader(splits["train"], root, transform, config["batch_size"], True)
    validation_loader = loader(splits["validate"], root, transform, config["batch_size"])
    history, epoch = train_head(
        model, train_loader, validation_loader, runtime, config["epochs"], config["patience"]
    )
    name = "resnet18_frozen"
    target = output / name
    target.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "best_epoch": epoch, "config": config},
        target / "best_model.pt",
    )
    pd.DataFrame(history).to_csv(target / "history.csv", index=False)
    _, predictions = predict_model(model, validation_loader, runtime)
    return {
        "validation": persist_evaluation(
            output, name, "validation", splits["validate"], predictions
        ),
        "best_epoch": epoch,
        "device": str(runtime),
    }


class TransformerTextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int) -> None:
        self.frame, self.tokenizer, self.max_length = (
            frame.reset_index(drop=True),
            tokenizer,
            max_length,
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            str(self.frame.iloc[index].text),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()} | {
            "labels": torch.tensor(int(self.frame.iloc[index].label)),
            "index": index,
        }


def text_transformer(
    splits: dict[str, pd.DataFrame], output: Path, config: dict[str, Any]
) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = config["text_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    runtime = device()
    model.to(runtime)
    train_loader = DataLoader(
        TransformerTextDataset(splits["train"], tokenizer, config["max_length"]),
        batch_size=config["text_batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TransformerTextDataset(splits["validate"], tokenizer, config["max_length"]),
        batch_size=config["text_batch_size"],
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-3)
    best, best_epoch, wait, state = -1.0, 0, 0, None
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        # The encoder is frozen: keep its dropout disabled. Besides being the
        # appropriate frozen-encoder protocol, this avoids MPS's unsupported
        # training-mode scaled-dot-product-attention dropout kernel.
        model.base_model.eval()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with amp_context(runtime):
                out = model(
                    input_ids=batch["input_ids"].to(runtime),
                    attention_mask=batch["attention_mask"].to(runtime),
                    labels=batch["labels"].to(runtime),
                )
            out.loss.backward()
            optimizer.step()
        labels, pred = predict_transformer(model, val_loader, runtime)
        current = f1_score(labels, pred, average="macro")
        if current > best:
            best, best_epoch, wait = current, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= config["patience"]:
                break
    model.load_state_dict(state)
    name = "pretrained_text_frozen"
    target = output / name
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target / "model")
    tokenizer.save_pretrained(target / "tokenizer")
    labels, pred = predict_transformer(model, val_loader, runtime)
    return {
        "validation": persist_evaluation(output, name, "validation", splits["validate"], pred),
        "best_epoch": best_epoch,
        "device": str(runtime),
        "model": model_name,
    }


def predict_transformer(model, data_loader, runtime):
    model.eval()
    labels = []
    pred = []
    with torch.no_grad():
        for batch in data_loader:
            with amp_context(runtime):
                logits = model(
                    input_ids=batch["input_ids"].to(runtime),
                    attention_mask=batch["attention_mask"].to(runtime),
                ).logits
            labels.extend(batch["labels"].tolist())
            pred.extend(logits.argmax(1).cpu().tolist())
    return np.asarray(labels), np.asarray(pred)


def clip_fusion(
    splits: dict[str, pd.DataFrame], root: Path, output: Path, config: dict[str, Any]
) -> dict[str, Any]:
    from transformers import CLIPModel, CLIPProcessor

    name = "clip_frozen_fusion"
    target = output / name
    target.mkdir(parents=True, exist_ok=True)
    model_name = config["clip_model"]
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device()).eval()
    runtime = device()

    def encode(frame):
        chunks = []
        for start in range(0, len(frame), config["clip_batch_size"]):
            part = frame.iloc[start : start + config["clip_batch_size"]]
            images = []
            for path in part.image_path:
                with Image.open(root / path) as im:
                    images.append(im.convert("RGB").copy())
            batch = processor(
                text=part.text.tolist(),
                images=images,
                padding=True,
                truncation=True,
                max_length=config["max_length"],
                return_tensors="pt",
            )
            with torch.no_grad(), amp_context(runtime):
                features = model.get_image_features(pixel_values=batch["pixel_values"].to(runtime))
                text = model.get_text_features(
                    input_ids=batch["input_ids"].to(runtime),
                    attention_mask=batch["attention_mask"].to(runtime),
                )
            chunks.append(torch.cat([features, text], dim=1).float().cpu().numpy())
        return np.vstack(chunks)

    vectors = {key: encode(value) for key, value in splits.items()}
    classifier = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced", random_state=config["seed"]
    )
    classifier.fit(vectors["train"], splits["train"].label)
    joblib.dump(classifier, target / "classifier.joblib")
    np.savez_compressed(
        target / "embeddings.npz",
        train=vectors["train"],
        validate=vectors["validate"],
        test=vectors["test"],
    )
    return {
        "validation": persist_evaluation(
            output, name, "validation", splits["validate"], classifier.predict(vectors["validate"])
        ),
        "model": model_name,
        "device": str(runtime),
    }


def write_comparison(
    output: Path, results: dict[str, Any], splits: dict[str, pd.DataFrame]
) -> None:
    rows = []
    for name, result in results.items():
        metrics = result["validation"]
        rows.append(
            {
                "model": name,
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
            }
        )
    pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False).to_csv(
        output / "comparison.csv", index=False
    )
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "split_counts": {key: len(value) for key, value in splits.items()},
                "label_mapping": {"0": "true", "1": "fake"},
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def evaluate_selected_test(
    best: str, splits: dict[str, pd.DataFrame], root: Path, output: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Touch test labels exactly once after the validation-only selection."""
    if best == "tfidf_svm":
        predictions = joblib.load(output / "tfidf_svm" / "model.joblib").predict(
            splits["test"].text
        )
    elif best == "resnet18_frozen":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(
            torch.load(
                output / "resnet18_frozen" / "best_model.pt", map_location="cpu", weights_only=True
            )["state_dict"]
        )
        _, predictions = predict_model(
            model.to(device()),
            loader(splits["test"], root, weights.transforms(), config["batch_size"]),
            device(),
        )
    elif best == "pretrained_text_frozen":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_path = output / "pretrained_text_frozen" / "model"
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device())
        tokenizer = AutoTokenizer.from_pretrained(output / "pretrained_text_frozen" / "tokenizer")
        _, predictions = predict_transformer(
            model,
            DataLoader(
                TransformerTextDataset(splits["test"], tokenizer, config["max_length"]),
                batch_size=config["text_batch_size"],
            ),
            device(),
        )
    elif best == "clip_frozen_fusion":
        predictions = joblib.load(output / "clip_frozen_fusion" / "classifier.joblib").predict(
            np.load(output / "clip_frozen_fusion" / "embeddings.npz")["test"]
        )
    else:
        raise ValueError(f"Unexpected selected model: {best}")
    return persist_evaluation(output, best, "test", splits["test"], predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/research_dataset_baselines.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    seed_everything(config["seed"])
    root = Path(config["dataset_root"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    splits = load_splits(root)
    results: dict[str, Any] = {}
    y = splits["validate"].label.to_numpy()
    majority = np.zeros_like(y)
    random_pred = np.random.default_rng(config["seed"]).choice(
        [0, 1],
        size=len(y),
        p=[splits["train"].label.eq(0).mean(), splits["train"].label.eq(1).mean()],
    )
    results["majority"] = {
        "validation": persist_evaluation(
            output, "majority", "validation", splits["validate"], majority
        )
    }
    results["random_stratified"] = {
        "validation": persist_evaluation(
            output, "random_stratified", "validation", splits["validate"], random_pred
        )
    }
    results["tfidf_svm"] = text_svm(splits, output)
    results["resnet18_frozen"] = resnet_baseline(splits, root, output, config)
    results["pretrained_text_frozen"] = text_transformer(splits, output, config)
    results["clip_frozen_fusion"] = clip_fusion(splits, root, output, config)
    write_comparison(output, results, splits)
    best = max(
        (name for name in results if name not in {"majority", "random_stratified"}),
        key=lambda name: results[name]["validation"]["macro_f1"],
    )
    # Test is evaluated once, after validation-only selection.
    results[best]["test"] = evaluate_selected_test(best, splits, root, output, config)
    write_comparison(output, results, splits)
    (output / "selected_model.json").write_text(
        json.dumps({"selected_by": "validation_macro_f1", "selected_model": best}, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_model": best,
                "validation": results[best]["validation"],
                "test": results[best]["test"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
