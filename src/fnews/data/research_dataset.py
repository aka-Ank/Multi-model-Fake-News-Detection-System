"""Build a bounded, provenance-preserving multimodal research dataset.

The builder intentionally keeps Fakeddit truth labels and NewsCLIPpings
image-caption mismatch labels in different columns.  NewsCLIPpings rows remain
metadata-only until both VisualNews metadata and the referenced image are
available locally; they are never presented as usable image-text pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import imagehash
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

GIB = 1024**3
SPLITS = ("train", "validate", "test")
UNIFIED_COLUMNS = [
    "sample_id",
    "source_dataset",
    "original_id",
    "image_id",
    "image_path",
    "text",
    "task_type",
    "original_label",
    "normalized_label",
    "fake_news_label",
    "image_caption_mismatch_label",
    "original_split",
    "final_split",
    "image_hash",
    "perceptual_hash",
    "download_status",
    "image_width",
    "image_height",
    "source_url",
    "dataset_version",
]


@dataclass(frozen=True)
class ProcessedImage:
    path: str
    sha256: str
    perceptual_hash: str
    width: int
    height: int
    bytes_written: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def _processed_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0


def _normalise_split(value: str) -> str:
    return "validate" if value == "val" else value


def _target_path(output_dir: Path, sample_id: str, image_format: str) -> Path:
    suffix = ".webp" if image_format.lower() == "webp" else ".jpg"
    return output_dir / f"{sample_id}{suffix}"


def validate_resize_encode(
    source: Path,
    destination: Path,
    *,
    max_dimension: int,
    image_format: str,
    quality: int,
) -> ProcessedImage:
    """Decode, orient, resize and atomically encode one image.

    Existing output is reused only after it decodes successfully, which makes
    interrupted acquisition safe to resume without trusting arbitrary bytes.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if destination.exists():
            with Image.open(destination) as existing:
                existing.verify()
            with Image.open(destination) as existing:
                rgb = ImageOps.exif_transpose(existing).convert("RGB")
                return ProcessedImage(
                    str(destination), _sha256(destination), str(imagehash.phash(rgb)), rgb.width, rgb.height,
                    destination.stat().st_size,
                )
        with Image.open(source) as image:
            image.verify()
        with Image.open(source) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            perceptual = str(imagehash.phash(rgb))
            temporary = destination.with_suffix(destination.suffix + ".part")
            save_format = "WEBP" if image_format.lower() == "webp" else "JPEG"
            kwargs: dict[str, Any] = {"format": save_format, "quality": quality}
            if save_format == "JPEG":
                kwargs.update({"optimize": True, "progressive": True})
            rgb.save(temporary, **kwargs)
            temporary.replace(destination)
            return ProcessedImage(
                str(destination), _sha256(destination), perceptual, rgb.width, rgb.height, destination.stat().st_size
            )
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError(f"{type(exc).__name__}: {exc}") from exc


def select_fakeddit_rows(frame: pd.DataFrame, per_split_cap: int, seed: int) -> pd.DataFrame:
    """Fairly select available Fakeddit samples within each official split."""
    valid = frame.loc[frame["valid_image"].fillna(False) & frame["local_path"].notna()].copy()
    valid["split"] = valid["split"].astype(str).map(_normalise_split)
    selected: list[pd.DataFrame] = []
    for split in SPLITS:
        group = valid.loc[valid["split"].eq(split)]
        if group.empty:
            continue
        classes = sorted(group["2_way_label"].dropna().astype(int).unique())
        quota = per_split_cap // max(1, len(classes))
        parts = [
            group.loc[group["2_way_label"].eq(label)].sample(
                n=min(quota, int(group["2_way_label"].eq(label).sum())), random_state=seed + label
            )
            for label in classes
        ]
        chosen = pd.concat(parts, ignore_index=True) if parts else group.iloc[0:0]
        if len(chosen) < min(per_split_cap, len(group)):
            remaining = group.loc[~group["id"].isin(chosen["id"])]
            chosen = pd.concat(
                [chosen, remaining.sample(n=min(per_split_cap - len(chosen), len(remaining)), random_state=seed + 101)],
                ignore_index=True,
            )
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True) if selected else valid.iloc[0:0]


def build_fakeddit(
    frame: pd.DataFrame,
    output_root: Path,
    *,
    cap_bytes: int,
    min_free_bytes: int,
    max_dimension: int,
    image_format: str,
    quality: int,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Materialise valid Fakeddit pairs without exceeding the image budget."""
    output_dir = output_root / "images" / "fakeddit"
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    assigned_hashes: dict[str, str] = {}
    assigned_phashes: list[tuple[str, str]] = []
    used_bytes = _processed_bytes(output_dir)
    for record in frame.sort_values(["split", "2_way_label", "id"]).to_dict("records"):
        if _free_bytes(output_root) < min_free_bytes:
            failures.append({"original_id": str(record["id"]), "reason": "minimum_free_space_reached"})
            continue
        split = _normalise_split(str(record["split"]))
        sample_id = f"fakeddit:{split}:{record['id']}"
        # Fakeddit IDs are not a safe filename namespace across split files.
        # Keeping them in split subdirectories prevents a later split from
        # overwriting a verified image from an earlier official partition.
        destination = _target_path(output_dir / split, str(record["id"]), image_format)
        prior_bytes = destination.stat().st_size if destination.exists() else 0
        try:
            image = validate_resize_encode(
                Path(record["local_path"]), destination, max_dimension=max_dimension, image_format=image_format, quality=quality
            )
        except ValueError as exc:
            failures.append({"original_id": str(record["id"]), "reason": str(exc)})
            continue
        if used_bytes - prior_bytes + image.bytes_written > cap_bytes:
            destination.unlink(missing_ok=True)
            failures.append({"original_id": str(record["id"]), "reason": "fakeddit_image_budget_reached"})
            continue
        # Preserve official split membership: conflicting images are excluded from
        # the leakage-controlled manifest instead of reassigned to another split.
        exact_split = assigned_hashes.get(image.sha256)
        near_splits = {prior_split for prior_hash, prior_split in assigned_phashes if _hamming(image.perceptual_hash, prior_hash) <= 2}
        if (exact_split and exact_split != split) or any(prior != split for prior in near_splits):
            if prior_bytes == 0:
                destination.unlink(missing_ok=True)
            failures.append({"original_id": str(record["id"]), "reason": "cross_split_duplicate_or_near_duplicate"})
            continue
        assigned_hashes[image.sha256] = split
        assigned_phashes.append((image.perceptual_hash, split))
        used_bytes = used_bytes - prior_bytes + image.bytes_written
        # Fakeddit's raw binary value 1 is its true class (six-way class 0).
        # Preserve that upstream value in ``original_label`` and store the
        # canonical task target as 1=non-true, 0=true for future FNEWS builds.
        source_label = int(record["2_way_label"])
        label = 1 - source_label
        rows.append(
            {
                "sample_id": sample_id,
                "source_dataset": "Fakeddit",
                "original_id": str(record["id"]),
                "image_id": str(record["id"]),
                "image_path": str(Path(image.path).relative_to(output_root)),
                "text": str(record["clean_title"]).strip(),
                "task_type": "fake_news_classification",
                "original_label": source_label,
                "normalized_label": "non_true" if label else "true",
                "fake_news_label": label,
                "image_caption_mismatch_label": pd.NA,
                "original_split": split,
                "final_split": split,
                "image_hash": image.sha256,
                "perceptual_hash": image.perceptual_hash,
                "download_status": "verified_reused_or_processed",
                "image_width": image.width,
                "image_height": image.height,
                "source_url": str(record["image_url"]),
                "dataset_version": "fnews-research-v1",
            }
        )
    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS), failures


def build_newsclippings_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve labels while making unresolved rows ineligible for loading."""
    result = pd.DataFrame(
        {
            "sample_id": "newsclippings:" + frame["sample_id"].astype(str),
            "source_dataset": "NewsCLIPpings",
            "original_id": frame["caption_id"].astype(str),
            "image_id": frame["image_id"].astype(str),
            "image_path": pd.NA,
            "text": pd.NA,
            "task_type": "image_caption_mismatch",
            "original_label": frame["falsified"].astype("Int8"),
            "normalized_label": frame["falsified"].map({0: "matched", 1: "mismatched"}),
            "fake_news_label": pd.NA,
            "image_caption_mismatch_label": frame["falsified"].astype("Int8"),
            "original_split": frame["original_split"].astype(str).map(_normalise_split),
            "final_split": frame["original_split"].astype(str).map(_normalise_split),
            "image_hash": pd.NA,
            "perceptual_hash": pd.NA,
            "download_status": "blocked_missing_visualnews_metadata_and_image",
            "image_width": pd.NA,
            "image_height": pd.NA,
            "source_url": pd.NA,
            "dataset_version": "fnews-research-v1",
        }
    )
    return result[UNIFIED_COLUMNS]


def _write_reports(root: Path, dataset: pd.DataFrame, failures: list[dict[str, str]], report_root: Path) -> dict[str, Any]:
    report_root.mkdir(parents=True, exist_ok=True)
    valid = dataset.loc[dataset["download_status"].eq("verified_reused_or_processed")]
    distributions = (
        dataset.groupby(["source_dataset", "task_type", "final_split", "normalized_label"], dropna=False)
        .size()
        .rename("samples")
        .reset_index()
    )
    distributions.to_csv(report_root / "class_distribution.csv", index=False)
    failures_frame = pd.DataFrame(failures, columns=["original_id", "reason"])
    log_dir = root / "download_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures_frame.to_csv(log_dir / "fakeddit_processing_failures.csv", index=False)
    exact_cross = valid.groupby("image_hash")["final_split"].nunique().gt(1).sum() if len(valid) else 0
    leakage = {
        "verified_fakeddit_pairs": int(len(valid)),
        "newsclippings_valid_pairs": 0,
        "cross_split_exact_hash_groups": int(exact_cross),
        "cross_split_near_duplicate_policy": "rejected_at_hamming_distance_lte_2",
        "rejected_rows": int(len(failures)),
        "failure_reasons": dict(Counter(item["reason"] for item in failures)),
    }
    (report_root / "leakage_report.json").write_text(json.dumps(leakage, indent=2, sort_keys=True), encoding="utf-8")
    storage = {
        "dataset_root": str(root),
        "final_dataset_bytes": _processed_bytes(root),
        "final_dataset_gib": round(_processed_bytes(root) / GIB, 4),
        "hard_cap_bytes": 20 * GIB,
        "fakeddit_image_target_bytes": 12 * GIB,
        "newsclippings_image_target_bytes": 6 * GIB,
        "metadata_target_bytes": GIB,
        "reserve_bytes": GIB,
        "free_bytes_at_report": _free_bytes(root),
    }
    (report_root / "storage_usage.json").write_text(json.dumps(storage, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "valid_paired_counts": {"Fakeddit": int(len(valid)), "NewsCLIPpings": 0},
        "blocked_reason": "VisualNews origin/data.json and referenced image corpus are unavailable locally; full archive was not downloaded.",
        "modality_coverage": {
            f"{source_dataset}:{status}": int(count)
            for (source_dataset, status), count in dataset.groupby(["source_dataset", "download_status"]).size().items()
        },
        "class_distribution_csv": str(report_root / "class_distribution.csv"),
        "storage_report": str(report_root / "storage_usage.json"),
    }
    (report_root / "dataset_quality_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (report_root / "dataset_acquisition.md").write_text(
        "# Bounded Dataset Acquisition\n\n"
        f"- Verified Fakeddit image-text pairs: **{len(valid):,}**.\n"
        "- Verified NewsCLIPpings image-caption pairs: **0**. Official annotations were reused, but VisualNews `origin/data.json` and the referenced images are absent. No VisualNews archive was downloaded.\n"
        "- `fake_news_label` and `image_caption_mismatch_label` are separate namespaces; mismatch is not claimed to be fake-news truthfulness.\n"
        f"- Rows rejected during validation/leakage control: {len(failures):,}.\n",
        encoding="utf-8",
    )
    return {**summary, **storage, **leakage}


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    if _free_bytes(root) < args.minimum_free_gib * GIB:
        raise RuntimeError("Insufficient free space before processing; no images were downloaded or transformed.")
    fakeddit = select_fakeddit_rows(pd.read_parquet(args.fakeddit_manifest), args.per_split_cap, args.seed)
    fakeddit_out, failures = build_fakeddit(
        fakeddit, root, cap_bytes=args.fakeddit_cap_gib * GIB, min_free_bytes=args.minimum_free_gib * GIB,
        max_dimension=args.max_dimension, image_format=args.image_format, quality=args.quality,
    )
    news = build_newsclippings_metadata(pd.read_parquet(args.newsclippings_manifest))
    unified = pd.concat([fakeddit_out, news], ignore_index=True)
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    unified.to_parquet(manifest_dir / "unified_manifest.parquet", index=False)
    for split in SPLITS:
        unified.loc[unified["final_split"].eq(split)].to_parquet(manifest_dir / f"{split}_manifest.parquet", index=False)
    report = _write_reports(root, unified, failures, root / "reports")
    version = {
        "dataset_version": "fnews-research-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "inputs": {"fakeddit_manifest": str(args.fakeddit_manifest), "newsclippings_manifest": str(args.newsclippings_manifest)},
        "result": report,
    }
    (root / "dataset_version_manifest.json").write_text(json.dumps(version, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a bounded, unified Fakeddit/NewsCLIPpings research dataset.")
    parser.add_argument("--fakeddit-manifest", default="data/processed/fakeddit/image_manifest.parquet")
    parser.add_argument("--newsclippings-manifest", default="data/processed/newsclippings/newsclippings_annotations_provisional.parquet")
    parser.add_argument("--output-root", default="data/research_dataset")
    parser.add_argument("--per-split-cap", type=int, default=2400)
    parser.add_argument("--fakeddit-cap-gib", type=int, default=12)
    parser.add_argument("--minimum-free-gib", type=int, default=2)
    parser.add_argument("--max-dimension", type=int, default=512)
    parser.add_argument("--image-format", choices=("jpeg", "webp"), default="jpeg")
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_dataset(parse_args()), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
