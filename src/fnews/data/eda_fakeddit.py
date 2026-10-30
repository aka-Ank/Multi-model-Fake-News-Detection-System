from __future__ import annotations

import argparse

import pandas as pd

from fnews.config import load_fakeddit_config
from fnews.data.fakeddit import discover_tsv_files, drop_unnamed_columns, normalize_bool_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Fakeddit EDA reports.")
    parser.add_argument("--config", default="configs/fakeddit.yaml", help="Path to Fakeddit config YAML.")
    return parser.parse_args()


def value_counts_table(frame: pd.DataFrame, column: str, context: dict[str, object]) -> pd.DataFrame:
    counts = frame[column].value_counts(dropna=False).rename_axis("value").reset_index(name="rows")
    for key, value in context.items():
        counts[key] = value
    return counts[[*context.keys(), "value", "rows"]]


def main() -> None:
    args = parse_args()
    config = load_fakeddit_config(args.config)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    dataset_summaries = []
    label_tables = []
    text_length_tables = []

    for split_file in discover_tsv_files(config):
        frame = pd.read_csv(split_file.path, sep="\t", low_memory=False)
        frame = drop_unnamed_columns(frame)
        context = {
            "source_file": str(split_file.path),
            "split": split_file.split,
            "is_multimodal_only_file": split_file.is_multimodal_only,
        }

        row = {**context, "rows": len(frame), "columns": len(frame.columns)}

        if config.text_column in frame.columns:
            text = frame[config.text_column].astype("string").fillna("")
            stripped = text.str.strip()
            lengths = stripped.str.len()
            row.update(
                {
                    "missing_text_rows": int(stripped.eq("").sum()),
                    "mean_text_chars": float(lengths.mean()),
                    "median_text_chars": float(lengths.median()),
                    "p95_text_chars": float(lengths.quantile(0.95)),
                }
            )
            text_length_tables.append(
                pd.DataFrame(
                    {
                        **context,
                        "text_chars": lengths,
                        "text_words": stripped.str.split().str.len(),
                    }
                )
            )

        if config.has_image_column in frame.columns:
            row["has_image_rows"] = int(normalize_bool_series(frame[config.has_image_column]).sum())

        if config.image_url_column in frame.columns:
            urls = frame[config.image_url_column].astype("string").fillna("").str.strip()
            row["rows_with_image_url"] = int(urls.ne("").sum())

        dataset_summaries.append(row)

        for label_column in config.label_columns:
            if label_column in frame.columns:
                label_tables.append(value_counts_table(frame, label_column, {**context, "label_column": label_column}))

    pd.DataFrame(dataset_summaries).to_csv(config.reports_dir / "dataset_summary.csv", index=False)

    if label_tables:
        pd.concat(label_tables, ignore_index=True).to_csv(config.reports_dir / "label_distribution.csv", index=False)

    if text_length_tables:
        pd.concat(text_length_tables, ignore_index=True).to_parquet(
            config.reports_dir / "text_length_observations.parquet",
            index=False,
        )

    print(f"Wrote EDA reports under {config.reports_dir}")


if __name__ == "__main__":
    main()
