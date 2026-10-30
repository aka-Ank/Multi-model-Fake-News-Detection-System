from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Fakeddit image research data and local hardware.")
    parser.add_argument(
        "--image-manifest",
        default="data/processed/fakeddit/image_manifest.parquet",
    )
    parser.add_argument(
        "--train-manifest",
        default="data/processed/fakeddit/multimodal_training_manifest.parquet",
    )
    parser.add_argument(
        "--validation-manifest",
        default="data/processed/fakeddit/multimodal_validation_manifest.parquet",
    )
    parser.add_argument("--output-dir", default="reports/image_research")
    return parser.parse_args()


def hardware_snapshot() -> dict[str, object]:
    snapshot: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or "unknown",
    }
    try:
        import torch

        snapshot.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "mps_available": bool(torch.backends.mps.is_available()),
            }
        )
    except ImportError:
        snapshot["torch_version"] = None
    try:
        memory_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        snapshot["memory_bytes"] = memory_bytes
    except (OSError, ValueError, subprocess.SubprocessError):
        snapshot["memory_bytes"] = None
    return snapshot


def image_integrity_summary(paths: pd.Series) -> tuple[dict[str, object], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for raw_path in paths.dropna().astype(str).drop_duplicates():
        path = Path(raw_path)
        if not path.is_file():
            rows.append({"path": raw_path, "status": "missing", "width": None, "height": None})
            continue
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                rows.append(
                    {
                        "path": raw_path,
                        "status": "valid",
                        "width": int(width),
                        "height": int(height),
                        "format": image.format,
                        "mode": image.mode,
                    }
                )
        except (OSError, UnidentifiedImageError) as error:
            rows.append({"path": raw_path, "status": "corrupt", "error": str(error)})
    frame = pd.DataFrame(rows)
    valid = frame[frame["status"] == "valid"].copy()
    summary: dict[str, object] = {
        "checked_files": int(len(frame)),
        "valid_files": int((frame["status"] == "valid").sum()),
        "missing_files": int((frame["status"] == "missing").sum()),
        "corrupt_files": int((frame["status"] == "corrupt").sum()),
    }
    if not valid.empty:
        pixels = valid["width"] * valid["height"]
        summary["width_quantiles"] = valid["width"].quantile([0.0, 0.5, 0.95, 1.0]).to_dict()
        summary["height_quantiles"] = valid["height"].quantile([0.0, 0.5, 0.95, 1.0]).to_dict()
        summary["pixel_quantiles"] = pixels.quantile([0.0, 0.5, 0.95, 1.0]).to_dict()
        summary["formats"] = valid["format"].value_counts(dropna=False).to_dict()
    return summary, frame


def duplicate_summary(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, int]:
    result: dict[str, int] = {}
    for column in ["id", "image_url", "sha256", "perceptual_hash"]:
        if column not in train or column not in validation:
            continue
        left = set(train[column].dropna().astype(str))
        right = set(validation[column].dropna().astype(str))
        result[f"cross_split_{column}_overlap"] = len(left & right)
    return result


def split_summary(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, group in frame.groupby("split", dropna=False):
        valid = group[group["valid_image"].fillna(False)]
        labels = group["2_way_label"].value_counts().to_dict()
        rows.append(
            {
                "split": str(split),
                "total_labeled_rows": int(len(group)),
                "valid_local_images": int(len(valid)),
                "image_coverage": float(len(valid) / len(group)) if len(group) else 0.0,
                "true_0": int(labels.get(0, 0)),
                "fake_1": int(labels.get(1, 0)),
            }
        )
    return rows


def run_audit(
    image_manifest_path: str | Path,
    train_manifest_path: str | Path,
    validation_manifest_path: str | Path,
    output_dir: str | Path,
) -> Path:
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_manifest = pd.read_parquet(image_manifest_path)
    train = pd.read_parquet(train_manifest_path)
    validation = pd.read_parquet(validation_manifest_path)
    local_records = pd.concat([train, validation], ignore_index=True)
    integrity, file_rows = image_integrity_summary(local_records["local_path"])
    duplicates = duplicate_summary(train, validation)
    coverage = split_summary(image_manifest)
    findings = {
        "runtime_seconds": time.perf_counter() - started,
        "hardware": hardware_snapshot(),
        "image_manifest_rows": int(len(image_manifest)),
        "coverage": coverage,
        "paired_train_rows": int(len(train)),
        "paired_validation_rows": int(len(validation)),
        "paired_train_class_counts": {str(key): int(value) for key, value in train["2_way_label"].value_counts().items()},
        "paired_validation_class_counts": {
            str(key): int(value) for key, value in validation["2_way_label"].value_counts().items()
        },
        "local_file_integrity": integrity,
        "cross_split_duplicates": duplicates,
        "acquisition_finding": (
            "Only locally available files appear in the paired manifests. Most image-manifest rows have no "
            "local file or recorded download error, so the missing corpus reflects incomplete acquisition rather "
            "than verified remote-image failures."
        ),
        "protocol": "Official test images are excluded from all training and validation selection.",
    }
    (output / "DATA_AUDIT.json").write_text(json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8")
    file_rows.to_csv(output / "image_integrity.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "split_coverage.csv", index=False)
    pd.DataFrame([duplicates]).to_csv(output / "cross_split_duplicates.csv", index=False)
    write_markdown(output / "DATA_AUDIT.md", findings)
    return output


def write_markdown(path: Path, findings: dict[str, object]) -> None:
    integrity = findings["local_file_integrity"]
    duplicates = findings["cross_split_duplicates"]
    lines = [
        "# Fakeddit Image Research Data Audit",
        "",
        "## Scope",
        "",
        "This audit uses the existing local Fakeddit image inventory and matched train/validation manifests. "
        "The official test split is not used for training, model selection, or metric reporting.",
        "",
        "## Coverage",
        "",
        pd.DataFrame(findings["coverage"]).to_markdown(index=False),
        "",
        "## Local File Integrity",
        "",
        pd.DataFrame([integrity]).to_markdown(index=False),
        "",
        "## Cross-Split Duplicate Risks",
        "",
        pd.DataFrame([duplicates]).to_markdown(index=False),
        "",
        "## Hardware",
        "",
        pd.DataFrame([findings["hardware"]]).to_markdown(index=False),
        "",
        "## Finding",
        "",
        str(findings["acquisition_finding"]),
        "",
        "## Protocol",
        "",
        str(findings["protocol"]),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = run_audit(args.image_manifest, args.train_manifest, args.validation_manifest, args.output_dir)
    print(f"Wrote image research audit under {output}")


if __name__ == "__main__":
    main()
