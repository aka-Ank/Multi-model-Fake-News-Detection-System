from __future__ import annotations

import argparse

import pandas as pd

from fnews.config import load_fakeddit_config
from fnews.data.fakeddit import (
    build_image_inventory,
    discover_tsv_files,
    drop_unnamed_columns,
    summarize_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Fakeddit manifests and image inventory.")
    parser.add_argument("--config", default="configs/fakeddit.yaml", help="Path to Fakeddit config YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_fakeddit_config(args.config)
    config.interim_dir.mkdir(parents=True, exist_ok=True)

    image_inventory = build_image_inventory(config.image_dir)
    image_inventory_path = config.interim_dir / "image_inventory.parquet"
    image_inventory.to_parquet(image_inventory_path, index=False)
    image_ids = set(image_inventory["image_id"].astype("string")) if not image_inventory.empty else set()

    summaries = []
    for split_file in discover_tsv_files(config):
        frame = pd.read_csv(split_file.path, sep="\t", low_memory=False)
        frame = drop_unnamed_columns(frame)
        summaries.append(
            summarize_frame(
                frame,
                source_path=split_file.path,
                split=split_file.split,
                is_multimodal_only=split_file.is_multimodal_only,
                config=config,
                image_ids=image_ids,
            )
        )

    summary_frame = pd.DataFrame(summaries)
    summary_path = config.interim_dir / "split_file_summary.csv"
    summary_frame.to_csv(summary_path, index=False)

    print(f"Wrote image inventory: {image_inventory_path}")
    print(f"Wrote split summary: {summary_path}")
    if summary_frame.empty:
        print(f"No TSV files found under {config.raw_dir}")


if __name__ == "__main__":
    main()
