from __future__ import annotations

import argparse

import pandas as pd

from fnews.config import load_fakeddit_config
from fnews.data.fakeddit import discover_tsv_files, drop_unnamed_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Fakeddit split leakage risks.")
    parser.add_argument("--config", default="configs/fakeddit.yaml", help="Path to Fakeddit config YAML.")
    return parser.parse_args()


def add_source_columns(frame: pd.DataFrame, source_file: str, split: str) -> pd.DataFrame:
    out = frame.copy()
    out["_source_file"] = source_file
    out["_split"] = split
    return out


def cross_split_duplicates(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    usable = frame[["_source_file", "_split", column]].dropna()
    usable[column] = usable[column].astype("string").str.strip()
    usable = usable[usable[column].ne("")]
    if usable.empty:
        return pd.DataFrame(columns=["column", "value", "splits", "rows", "source_files"])

    grouped = (
        usable.groupby(column)
        .agg(
            splits=("_split", lambda values: sorted(set(values))),
            rows=("_split", "size"),
            source_files=("_source_file", lambda values: sorted(set(values))),
        )
        .reset_index()
    )
    grouped = grouped[grouped["splits"].map(len) > 1]
    grouped.insert(0, "column", column)
    grouped = grouped.rename(columns={column: "value"})
    return grouped


def main() -> None:
    args = parse_args()
    config = load_fakeddit_config(args.config)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for split_file in discover_tsv_files(config):
        frame = pd.read_csv(split_file.path, sep="\t", low_memory=False)
        frame = drop_unnamed_columns(frame)
        frames.append(add_source_columns(frame, str(split_file.path), split_file.split))

    if not frames:
        print(f"No TSV files found under {config.raw_dir}")
        return

    combined = pd.concat(frames, ignore_index=True, sort=False)
    candidate_columns = [
        config.id_column,
        config.text_column,
        config.image_url_column,
        "url",
        "domain",
        "author",
        "subreddit",
    ]
    available_columns = [column for column in candidate_columns if column in combined.columns]

    duplicate_tables = [cross_split_duplicates(combined, column) for column in available_columns]
    duplicate_tables = [table for table in duplicate_tables if not table.empty]

    output_path = config.reports_dir / "cross_split_duplicates.csv"
    if duplicate_tables:
        pd.concat(duplicate_tables, ignore_index=True).to_csv(output_path, index=False)
    else:
        pd.DataFrame(columns=["column", "value", "splits", "rows", "source_files"]).to_csv(output_path, index=False)

    within_split_path = config.reports_dir / "within_split_duplicate_ids.csv"
    if config.id_column in combined.columns:
        id_frame = combined[["_source_file", "_split", config.id_column]].copy()
        id_frame[config.id_column] = id_frame[config.id_column].astype("string").str.strip()
        duplicates = id_frame[id_frame.duplicated(["_source_file", config.id_column], keep=False)]
        duplicates.to_csv(within_split_path, index=False)

    print(f"Wrote leakage report: {output_path}")
    if config.id_column in combined.columns:
        print(f"Wrote duplicate ID report: {within_split_path}")


if __name__ == "__main__":
    main()
