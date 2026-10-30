from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fnews.data.newsclippings.acquisition import acquire_annotations, free_gib
from fnews.data.newsclippings.manifest import build_provisional_manifest, split_overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire official NewsCLIPpings annotations and build a provisional manifest.")
    parser.add_argument("--root", default="data/raw/newsclippings")
    parser.add_argument("--report-dir", default="reports/datasets")
    parser.add_argument("--output-manifest", default="data/processed/newsclippings/newsclippings_annotations_provisional.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    report_dir = Path(args.report_dir)
    records = acquire_annotations(root, report_dir / "newsclippings_download_log.jsonl")
    manifest = build_provisional_manifest(root, Path(args.output_manifest))
    report_dir.mkdir(parents=True, exist_ok=True)
    overlaps = pd.concat([split_overlap(manifest, column) for column in ("caption_id", "image_id")], ignore_index=True)
    overlaps.to_csv(report_dir / "newsclippings_split_overlaps.csv", index=False)
    status = pd.DataFrame([record.__dict__ for record in records])
    summary = {
        "downloaded_bytes": int(status.loc[status.status.isin(["downloaded", "exists"]), "bytes_downloaded"].sum()),
        "annotation_files": len(status),
        "annotation_failures": int((status.status == "failed").sum()),
        "annotations": len(manifest),
        "valid_labeled_pairs": 0,
        "free_gib_after": round(free_gib(root), 2),
    }
    (report_dir / "newsclippings_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (report_dir / "NEWSCLIPPINGS_ACQUISITION.md").write_text(
        "# NewsCLIPpings Acquisition\n\n"
        f"- Annotation files: {summary['annotation_files']}\n"
        f"- Downloaded or reused bytes: {summary['downloaded_bytes']:,}\n"
        f"- Annotation rows: {summary['annotations']:,}\n"
        "- Verified image-caption pairs: 0; official VisualNews origin images and data.json are not available locally.\n"
        "- This manifest preserves NewsCLIPpings falsified labels, which mean generated image-caption mismatch, not fake-news truthfulness.\n",
        encoding="utf-8",
    )
    (report_dir / "NEWSCLIPPINGS_READINESS.md").write_text(
        "# NewsCLIPpings Readiness\n\n"
        "Annotations are available and schema-validated. Training cannot begin safely because VisualNews `origin/data.json` and "
        "the referenced images are absent. The official origin archive is 91 GB, which exceeds the configured approval threshold "
        "and available local storage.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
