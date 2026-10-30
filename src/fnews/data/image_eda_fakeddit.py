from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imagehash
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from fnews.config import load_fakeddit_config
from fnews.data.fakeddit import IMAGE_SUFFIXES, discover_tsv_files, drop_unnamed_columns

SPLIT_ORDER = ["train", "validate", "test"]


@dataclass(frozen=True)
class ImageInspection:
    id: str
    local_path: str | None
    downloaded: bool
    valid_image: bool
    file_extension: str | None
    image_format: str | None
    image_mode: str | None
    width: int | None
    height: int | None
    aspect_ratio: float | None
    file_size_bytes: int | None
    file_size_kb: float | None
    pixel_count: int | None
    sha256: str | None
    perceptual_hash: str | None
    error_type: str | None
    error_message: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Fakeddit local image EDA and leakage checks.")
    parser.add_argument("--config", default="configs/fakeddit.yaml", help="Path to Fakeddit config YAML.")
    parser.add_argument("--split", choices=[*SPLIT_ORDER, "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for fast pilot runs.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel image inspection workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample image grids.")
    parser.add_argument("--force", action="store_true", help="Recompute even if cached outputs exist.")
    return parser.parse_args()


def find_local_images(image_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    if not image_dir.exists():
        return images
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        images.setdefault(path.stem, path)
    return images


def calculate_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(image_id: str, path: Path | None) -> ImageInspection:
    if path is None:
        return ImageInspection(
            id=image_id,
            local_path=None,
            downloaded=False,
            valid_image=False,
            file_extension=None,
            image_format=None,
            image_mode=None,
            width=None,
            height=None,
            aspect_ratio=None,
            file_size_bytes=None,
            file_size_kb=None,
            pixel_count=None,
            sha256=None,
            perceptual_hash=None,
            error_type=None,
            error_message=None,
        )

    suffix = path.suffix.lower()
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        return _failure(image_id, path, suffix, "stat_error", str(exc))

    if file_size == 0:
        return _failure(image_id, path, suffix, "zero_byte", "File is empty", file_size)

    try:
        with path.open("rb") as handle:
            header = handle.read(512).lstrip().lower()
        if header.startswith((b"<!doctype html", b"<html")):
            return _failure(image_id, path, suffix, "html_response", "File starts with HTML", file_size)

        sha256 = calculate_sha256(path)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format
            image_mode = image.mode
            perceptual_hash = str(imagehash.phash(image.convert("RGB")))
    except UnidentifiedImageError as exc:
        return _failure(image_id, path, suffix, "unidentified_image", str(exc), file_size)
    except OSError as exc:
        return _failure(image_id, path, suffix, "corrupt_or_truncated", str(exc), file_size)
    except ValueError as exc:
        return _failure(image_id, path, suffix, "unsupported_image", str(exc), file_size)

    pixel_count = width * height
    return ImageInspection(
        id=image_id,
        local_path=str(path),
        downloaded=True,
        valid_image=True,
        file_extension=suffix,
        image_format=image_format,
        image_mode=image_mode,
        width=width,
        height=height,
        aspect_ratio=width / height if height else None,
        file_size_bytes=file_size,
        file_size_kb=file_size / 1024,
        pixel_count=pixel_count,
        sha256=sha256,
        perceptual_hash=perceptual_hash,
        error_type=None,
        error_message=None,
    )


def _failure(
    image_id: str,
    path: Path,
    suffix: str,
    error_type: str,
    error_message: str,
    file_size: int | None = None,
) -> ImageInspection:
    return ImageInspection(
        id=image_id,
        local_path=str(path),
        downloaded=True,
        valid_image=False,
        file_extension=suffix,
        image_format=None,
        image_mode=None,
        width=None,
        height=None,
        aspect_ratio=None,
        file_size_bytes=file_size,
        file_size_kb=(file_size / 1024 if file_size is not None else None),
        pixel_count=None,
        sha256=None,
        perceptual_hash=None,
        error_type=error_type,
        error_message=error_message,
    )


def load_multimodal_rows(config_path: str, split_filter: str = "all", limit: int | None = None) -> pd.DataFrame:
    config = load_fakeddit_config(config_path)
    frames = []
    for split_file in discover_tsv_files(config):
        if not split_file.is_multimodal_only:
            continue
        if split_filter != "all" and split_file.split != split_filter:
            continue
        frame = pd.read_csv(split_file.path, sep="\t", low_memory=False)
        frame = drop_unnamed_columns(frame)
        keep = [
            column
            for column in [
                "id",
                "split",
                "clean_title",
                "2_way_label",
                "3_way_label",
                "6_way_label",
                "image_url",
            ]
            if column in frame.columns or column == "split"
        ]
        frame["split"] = split_file.split
        frame = frame[keep]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["split"] = pd.Categorical(combined["split"], SPLIT_ORDER, ordered=True)
    combined = combined.sort_values(["split", "id"]).reset_index(drop=True)
    if limit is not None:
        combined = combined.head(limit)
    return combined


def build_image_manifest(config_path: str, split: str = "all", limit: int | None = None, workers: int = 4) -> pd.DataFrame:
    config = load_fakeddit_config(config_path)
    rows = load_multimodal_rows(config_path, split, limit)
    local_images = find_local_images(config.image_dir)
    if rows.empty:
        return rows

    requested = [(str(image_id), local_images.get(str(image_id))) for image_id in rows["id"].astype("string")]
    inspections: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(inspect_image, image_id, path) for image_id, path in requested]
        for future in tqdm(as_completed(futures), total=len(futures), desc="inspect images"):
            inspection = future.result()
            inspections[inspection.id] = asdict(inspection)

    inspection_frame = pd.DataFrame(inspections.values())
    manifest = rows.merge(inspection_frame, on="id", how="left")
    manifest["downloaded"] = manifest["downloaded"].fillna(False).astype(bool)
    manifest["valid_image"] = manifest["valid_image"].fillna(False).astype(bool)
    return manifest


def summarize_coverage(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in manifest.groupby("split", observed=True):
        expected = len(group)
        downloaded = int(group["downloaded"].sum())
        valid = int(group["valid_image"].sum())
        corrupt = int((group["downloaded"] & ~group["valid_image"]).sum())
        image_url_rows = int(group["image_url"].astype("string").fillna("").str.strip().ne("").sum())
        rows.append(
            {
                "split": split,
                "expected_multimodal_rows": expected,
                "rows_with_image_url": image_url_rows,
                "locally_downloaded_images": downloaded,
                "valid_images": valid,
                "missing_images": expected - downloaded,
                "corrupted_images": corrupt,
                "coverage_pct": downloaded / expected * 100 if expected else 0,
                "valid_coverage_pct": valid / expected * 100 if expected else 0,
            }
        )
    return pd.DataFrame(rows)


def summarize_availability_by_label(manifest: pd.DataFrame) -> pd.DataFrame:
    if "2_way_label" not in manifest.columns:
        return pd.DataFrame()
    rows = []
    for (split, label), group in manifest.groupby(["split", "2_way_label"], observed=True):
        expected = len(group)
        downloaded = int(group["downloaded"].sum())
        valid = int(group["valid_image"].sum())
        rows.append(
            {
                "split": split,
                "2_way_label": label,
                "expected": expected,
                "downloaded": downloaded,
                "valid": valid,
                "missing": expected - downloaded,
                "download_coverage_pct": downloaded / expected * 100 if expected else 0,
                "valid_coverage_pct": valid / expected * 100 if expected else 0,
            }
        )
    return pd.DataFrame(rows)


def image_property_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    valid = manifest[manifest["valid_image"]].copy()
    numeric_columns = ["width", "height", "aspect_ratio", "pixel_count", "file_size_kb"]
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.groupby(["split", "2_way_label"], observed=True)[numeric_columns]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
        .reset_index()
    )


def exact_duplicates(manifest: pd.DataFrame) -> pd.DataFrame:
    return duplicate_groups(manifest, "sha256")


def perceptual_duplicates(manifest: pd.DataFrame) -> pd.DataFrame:
    return duplicate_groups(manifest, "perceptual_hash")


def duplicate_groups(manifest: pd.DataFrame, column: str) -> pd.DataFrame:
    valid = manifest[manifest["valid_image"] & manifest[column].notna()].copy()
    if valid.empty:
        return pd.DataFrame(columns=[column, "sample_count", "ids", "splits", "labels", "multiple_labels"])
    grouped = (
        valid.groupby(column)
        .agg(
            sample_count=("id", "size"),
            ids=("id", lambda values: ",".join(map(str, values))),
            splits=("split", lambda values: ",".join(sorted(set(map(str, values))))),
            labels=("2_way_label", lambda values: ",".join(map(str, sorted(set(values))))),
        )
        .reset_index()
    )
    grouped = grouped[grouped["sample_count"] > 1].sort_values("sample_count", ascending=False)
    grouped["multiple_labels"] = grouped["labels"].str.contains(",")
    return grouped


def leakage_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    methods = [
        ("image_url", "URL reuse"),
        ("sha256", "exact-byte duplication"),
        ("perceptual_hash", "perceptual-hash equality"),
    ]
    pairs = [("train", "validate"), ("train", "test"), ("validate", "test")]
    rows = []
    for column, method in methods:
        values_by_split = {}
        for split in SPLIT_ORDER:
            series = manifest.loc[manifest["split"].astype(str) == split, column].astype("string").fillna("").str.strip()
            values_by_split[split] = set(series[series.ne("")])
        row = {"method": method}
        for left, right in pairs:
            row[f"{left}-{right}"] = len(values_by_split[left] & values_by_split[right])
        rows.append(row)
    return pd.DataFrame(rows)


def image_text_reuse(manifest: pd.DataFrame) -> pd.DataFrame:
    valid = manifest[manifest["valid_image"]].copy()
    rows = []
    if not valid.empty:
        image_title = (
            valid.groupby("sha256")
            .agg(
                image_count=("id", "size"),
                unique_titles=("clean_title", "nunique"),
                unique_labels=("2_way_label", "nunique"),
                ids=("id", lambda values: ",".join(map(str, values))),
                splits=("split", lambda values: ",".join(sorted(set(map(str, values))))),
            )
            .reset_index()
        )
        image_title = image_title[(image_title["image_count"] > 1) | (image_title["unique_titles"] > 1)]
        image_title.insert(0, "pattern", "same_image_reuse")
        rows.append(image_title.head(100))

        title_images = (
            valid.groupby(valid["clean_title"].astype("string").str.strip().str.lower())
            .agg(
                image_count=("sha256", "nunique"),
                unique_labels=("2_way_label", "nunique"),
                ids=("id", lambda values: ",".join(map(str, values))),
                splits=("split", lambda values: ",".join(sorted(set(map(str, values))))),
            )
            .reset_index(names="clean_title_norm")
        )
        title_images = title_images[title_images["image_count"] > 1].sort_values("image_count", ascending=False)
        title_images.insert(0, "pattern", "same_title_multiple_images")
        rows.append(title_images.head(100))
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def validation_failures(manifest: pd.DataFrame) -> pd.DataFrame:
    failures = manifest[manifest["downloaded"] & ~manifest["valid_image"]].copy()
    columns = ["id", "split", "local_path", "image_url", "error_type", "error_message"]
    return failures[[column for column in columns if column in failures.columns]].rename(columns={"local_path": "path"})


def plot_image_eda(manifest: pd.DataFrame, output_dir: Path, seed: int = 42) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = manifest[manifest["valid_image"]].copy()
    if valid.empty:
        return

    for column, title, filename, log_scale in [
        ("width", "Image width distribution", "width_distribution.png", True),
        ("height", "Image height distribution", "height_distribution.png", True),
        ("aspect_ratio", "Image aspect ratio distribution", "aspect_ratio_distribution.png", False),
        ("pixel_count", "Pixel count distribution", "pixel_count_distribution.png", True),
        ("file_size_kb", "File size distribution", "file_size_distribution.png", True),
    ]:
        plt.figure(figsize=(9, 5))
        sns.histplot(valid[column].dropna(), bins=min(40, max(5, len(valid))))
        if log_scale:
            plt.xscale("log")
        plt.title(title)
        plt.xlabel(column)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=valid, x="image_format", order=valid["image_format"].value_counts().index)
    plt.title("Image formats")
    plt.xlabel("")
    plt.ylabel("Images")
    plt.tight_layout()
    plt.savefig(output_dir / "format_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=valid, x="image_mode", order=valid["image_mode"].value_counts().index)
    plt.title("Pillow color modes")
    plt.xlabel("")
    plt.ylabel("Images")
    plt.tight_layout()
    plt.savefig(output_dir / "mode_distribution.png", dpi=160)
    plt.close()

    if "2_way_label" in valid.columns and valid["2_way_label"].nunique() > 1:
        for column, filename in [
            ("width", "width_by_label.png"),
            ("height", "height_by_label.png"),
            ("aspect_ratio", "aspect_ratio_by_label.png"),
            ("pixel_count", "pixel_count_by_label.png"),
            ("file_size_kb", "file_size_by_label.png"),
        ]:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=valid, x="2_way_label", y=column)
            if column in {"pixel_count", "file_size_kb"}:
                plt.yscale("log")
            plt.title(f"{column} by 2-way label")
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=160)
            plt.close()

    make_sample_grids(valid, output_dir, seed)


def make_sample_grids(valid: pd.DataFrame, output_dir: Path, seed: int) -> None:
    grid_specs = {
        "random_true_examples.png": valid[valid["2_way_label"] == 0].sample(min(6, int((valid["2_way_label"] == 0).sum())), random_state=seed),
        "random_fake_examples.png": valid[valid["2_way_label"] == 1].sample(min(6, int((valid["2_way_label"] == 1).sum())), random_state=seed),
        "smallest_images.png": valid.nsmallest(min(6, len(valid)), "pixel_count"),
        "largest_images.png": valid.nlargest(min(6, len(valid)), "pixel_count"),
        "unusual_aspect_ratios.png": valid.assign(aspect_distance=(valid["aspect_ratio"] - 1).abs()).nlargest(min(6, len(valid)), "aspect_distance"),
    }
    dupes = exact_duplicates(valid)
    if not dupes.empty:
        dupe_ids = str(dupes.iloc[0]["ids"]).split(",")
        grid_specs["repeated_image_examples.png"] = valid[valid["id"].astype(str).isin(dupe_ids)].head(6)

    for filename, sample in grid_specs.items():
        if sample.empty:
            continue
        save_image_grid(sample, output_dir / filename)


def save_image_grid(sample: pd.DataFrame, output_path: Path) -> None:
    cols = 3
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for axis in axes_list:
        axis.axis("off")
    for axis, (_, row) in zip(axes_list, sample.iterrows(), strict=False):
        try:
            with Image.open(row["local_path"]) as image:
                axis.imshow(image.convert("RGB"))
        except (OSError, ValueError):
            continue
        axis.set_title(
            f"id={row['id']}\nlabel={row.get('2_way_label', '')} | {int(row['width'])}x{int(row['height'])}",
            fontsize=9,
        )
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def write_image_report(output_path: Path, manifest: pd.DataFrame, coverage: pd.DataFrame, leakage: pd.DataFrame) -> None:
    valid = manifest[manifest["valid_image"]].copy()
    downloaded = int(manifest["downloaded"].sum())
    valid_count = int(manifest["valid_image"].sum())
    corrupt_count = int((manifest["downloaded"] & ~manifest["valid_image"]).sum())
    exact_dupes = exact_duplicates(manifest)
    perceptual_dupes = perceptual_duplicates(manifest)

    lines = [
        "# Fakeddit Image EDA",
        "",
        "## Summary",
        "",
        f"- Expected multimodal rows scanned: {len(manifest):,}",
        f"- Local image files matched by Fakeddit id: {downloaded:,}",
        f"- Valid local images: {valid_count:,}",
        f"- Corrupted/invalid local images: {corrupt_count:,}",
        f"- Missing local images: {len(manifest) - downloaded:,}",
        "",
        "## Local Image Coverage",
        "",
        coverage.to_markdown(index=False),
        "",
        "## Cross-Split Image Leakage",
        "",
        leakage.to_markdown(index=False),
        "",
        "## Duplicate Images",
        "",
        f"- Exact SHA-256 duplicate groups: {len(exact_dupes):,}",
        f"- Exact perceptual-hash duplicate groups: {len(perceptual_dupes):,}",
        "",
    ]
    if not valid.empty:
        desc = valid[["width", "height", "aspect_ratio", "pixel_count", "file_size_kb"]].describe(
            percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]
        )
        lines.extend(["## Resolution and File Size", "", desc.to_markdown(), ""])
        lines.extend(
            [
                "## Dominant Formats",
                "",
                valid["image_format"].value_counts().to_frame("images").to_markdown(),
                "",
                "## Shortcut-Learning Notes",
                "",
                "Resolution, file size, color mode, and format are non-semantic signals. If they differ by label after larger image downloads, image-only or multimodal models may exploit collection artifacts instead of misinformation evidence.",
                "",
            ]
        )
    else:
        lines.append("No valid local images were available for property analysis.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_image_eda(
    config_path: str,
    split: str = "all",
    limit: int | None = None,
    workers: int = 4,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Path]:
    config = load_fakeddit_config(config_path)
    report_dir = config.reports_dir / "deep_eda"
    figure_dir = report_dir / "figures" / "images"
    processed_dir = config.raw_dir.parent / "processed" / "fakeddit"
    # Project convention is data/processed, not data/raw/../processed.
    processed_dir = Path(config_path).resolve().parent.parent / "data" / "processed" / "fakeddit"
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = processed_dir / "image_manifest.parquet"
    pilot_manifest_path = processed_dir / "image_manifest_pilot.parquet"
    active_manifest_path = manifest_path if limit is None and split == "all" else pilot_manifest_path
    if active_manifest_path.exists() and not force:
        manifest = pd.read_parquet(active_manifest_path)
    else:
        manifest = build_image_manifest(config_path, split=split, limit=limit, workers=workers)
        manifest.to_parquet(active_manifest_path, index=False)

    dimensions_path = report_dir / "image_dimensions.parquet"
    manifest[manifest["valid_image"]].to_parquet(dimensions_path, index=False)

    coverage = summarize_coverage(manifest)
    label_coverage = summarize_availability_by_label(manifest)
    properties = image_property_summary(manifest)
    failures = validation_failures(manifest)
    exact = exact_duplicates(manifest)
    perceptual = perceptual_duplicates(manifest)
    leakage = leakage_summary(manifest)
    reuse = image_text_reuse(manifest)

    coverage.to_csv(report_dir / "image_summary.csv", index=False)
    label_coverage.to_csv(report_dir / "image_availability_by_label.csv", index=False)
    properties.to_csv(report_dir / "image_label_property_summary.csv", index=False)
    failures.to_csv(report_dir / "image_validation_failures.csv", index=False)
    exact.to_csv(report_dir / "exact_image_duplicates.csv", index=False)
    perceptual.to_csv(report_dir / "perceptual_image_duplicates.csv", index=False)
    leakage.to_csv(report_dir / "image_leakage_summary.csv", index=False)
    reuse.to_csv(report_dir / "image_text_reuse_patterns.csv", index=False)

    plot_image_eda(manifest, figure_dir, seed=seed)
    write_image_report(report_dir / "image_eda_report.md", manifest, coverage, leakage)
    return {
        "manifest": manifest_path,
        "report_dir": report_dir,
        "figure_dir": figure_dir,
    }


def main() -> None:
    args = parse_args()
    outputs = run_image_eda(
        args.config,
        split=args.split,
        limit=args.limit,
        workers=args.workers,
        seed=args.seed,
        force=args.force,
    )
    print(f"Wrote image EDA outputs under {outputs['report_dir']}")


if __name__ == "__main__":
    main()
