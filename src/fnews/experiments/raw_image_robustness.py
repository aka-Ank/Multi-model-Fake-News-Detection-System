"""Resumable raw-image robustness sweep for the frozen gated-fusion checkpoint."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, f1_score
from transformers import CLIPModel, CLIPProcessor

from fnews.ml.models.clip_fusion import GatedFusionHead

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "reports/phase2/raw_image_robustness"
CONDITIONS = (
    "original",
    "jpeg_q95", "jpeg_q75", "jpeg_q50", "jpeg_q25",
    "resize_384", "resize_256", "resize_128",
    "blur_0.5", "blur_1", "blur_2", "blur_3",
    "brightness_0.7", "brightness_0.85", "brightness_1.15", "brightness_1.3",
    "occlusion_10", "occlusion_20", "occlusion_30",
)


def transform(im: Image.Image, name: str, seed: int) -> Image.Image:
    """Apply one deterministic degradation, maintaining an RGB image."""
    im = im.convert("RGB")
    if name == "original":
        return im
    if name.startswith("jpeg_q"):
        quality = int(name.removeprefix("jpeg_q"))
        encoded = io.BytesIO()
        im.save(encoded, format="JPEG", quality=quality)
        return Image.open(io.BytesIO(encoded.getvalue())).convert("RGB")
    if name.startswith("resize_"):
        dimension = int(name.removeprefix("resize_"))
        im.thumbnail((dimension, dimension), Image.Resampling.LANCZOS)
        return im
    if name.startswith("blur_"):
        return im.filter(ImageFilter.GaussianBlur(float(name.removeprefix("blur_"))))
    if name.startswith("brightness_"):
        return ImageEnhance.Brightness(im).enhance(float(name.removeprefix("brightness_")))
    if name.startswith("occlusion_"):
        fraction = int(name.removeprefix("occlusion_")) / 100
        width, height = im.size
        generator = np.random.default_rng(seed)
        side = int((width * height * fraction) ** 0.5)
        x = int(generator.integers(0, max(1, width - side + 1)))
        y = int(generator.integers(0, max(1, height - side + 1)))
        ImageDraw.Draw(im).rectangle((x, y, x + side, y + side), fill=(0, 0, 0))
        return im
    raise ValueError(f"Unknown transformation: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    base = data[data.transformation.eq("original")].set_index("sample_id")
    rows = []
    for name, group in data.groupby("transformation", sort=False):
        aligned = group.set_index("sample_id").loc[base.index]
        rows.append({
            "transformation": name,
            "samples": len(aligned),
            "accuracy": float(accuracy_score(aligned.label, aligned.prediction)),
            "macro_f1": float(f1_score(aligned.label, aligned.prediction, average="macro", zero_division=0)),
            "agreement_vs_original": float((aligned.prediction == base.prediction).mean()),
            "mean_abs_probability_shift": float(np.abs(aligned.raw_fake_probability - base.raw_fake_probability).mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifests = pd.read_parquet(ROOT / "data/research_dataset/manifests/validate_manifest.parquet")
    frame = manifests[(manifests.source_dataset == "Fakeddit") & (manifests.download_status == "verified_reused_or_processed")]
    sample = pd.concat([
        frame[frame.fake_news_label.eq(label)].sample(args.per_class, random_state=args.seed + label)
        for label in (0, 1)
    ]).reset_index(drop=True)
    results_path = output / "results.parquet"
    data = pd.read_parquet(results_path) if results_path.exists() else pd.DataFrame()
    completed = set(data.transformation.unique()) if not data.empty else set()

    checkpoint = torch.load(ROOT / "reports/multimodal_fusion/clip_fusion_v1/gated/best_model.pt", map_location="cpu", weights_only=False)
    head = GatedFusionHead(hidden_dim=256, dropout=0.2)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    mean, std = checkpoint["feature_mean"], checkpoint["feature_std"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    for name in CONDITIONS:
        if name in completed:
            continue
        images = []
        for index, row in sample.iterrows():
            with Image.open(ROOT / "data/research_dataset" / row.image_path) as opened:
                images.append(transform(opened.copy(), name, args.seed + index))
        batch = processor(text=sample.text.tolist(), images=images, padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            image_features = clip.get_image_features(pixel_values=batch.pixel_values.to(device)).cpu().numpy()
            text_features = clip.get_text_features(input_ids=batch.input_ids.to(device), attention_mask=batch.attention_mask.to(device)).cpu().numpy()
            feature = (np.c_[image_features, text_features] - mean) / std
            logits = head(torch.from_numpy(feature[:, :512]).float(), torch.from_numpy(feature[:, 512:]).float())
            probability = torch.softmax(logits, 1)[:, 1].numpy()
            prediction = logits.argmax(1).numpy()
        condition = pd.DataFrame({
            "sample_id": sample.sample_id,
            "label": sample.fake_news_label.astype(int),
            "transformation": name,
            "prediction": prediction.astype(int),
            "raw_fake_probability": probability.astype(float),
            "seed": [args.seed + index for index in range(len(sample))],
        })
        data = pd.concat([data, condition], ignore_index=True)
        data.to_parquet(results_path, index=False)
        print(f"persisted {name}: {len(data)} rows")

    summary = summarize(data)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "config.json").write_text(json.dumps({
        "samples": len(sample), "per_class": args.per_class, "seed": args.seed,
        "conditions": list(CONDITIONS), "raw_image_reencoded": True,
        "clip_text_max_length": 64,
        "resumable": "results.parquet is persisted after every condition",
        "frozen_checkpoint": "reports/multimodal_fusion/clip_fusion_v1/gated/best_model.pt",
    }, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
