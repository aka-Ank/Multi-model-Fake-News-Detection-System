from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fnews.config import load_fakeddit_config
from fnews.data.fakeddit import discover_tsv_files, drop_unnamed_columns, normalize_bool_series

LABEL_NAMES = {
    "2_way_label": {0: "true", 1: "fake"},
    "3_way_label": {0: "true", 1: "partially false", 2: "false"},
    "6_way_label": {
        0: "true",
        1: "satire/parody",
        2: "misleading",
        3: "imposter",
        4: "false connection",
        5: "manipulated content",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a detailed Fakeddit EDA report.")
    parser.add_argument("--config", default="configs/fakeddit.yaml", help="Path to Fakeddit config YAML.")
    return parser.parse_args()


def read_split_frames(config_path: str) -> list[tuple[str, str, bool, pd.DataFrame]]:
    config = load_fakeddit_config(config_path)
    frames = []
    for split_file in discover_tsv_files(config):
        frame = pd.read_csv(split_file.path, sep="\t", low_memory=False)
        frame = drop_unnamed_columns(frame)
        frames.append((str(split_file.path), split_file.split, split_file.is_multimodal_only, frame))
    return frames


def subset_name(is_multimodal_only: bool) -> str:
    return "multimodal_only" if is_multimodal_only else "all_samples"


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def quantile_summary(series: pd.Series, prefix: str) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    return {
        f"{prefix}_mean": float(numeric.mean()),
        f"{prefix}_median": float(numeric.median()),
        f"{prefix}_p95": float(numeric.quantile(0.95)),
        f"{prefix}_max": float(numeric.max()),
    }


def build_dataset_summary(frames: list[tuple[str, str, bool, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for source_file, split, is_mm, frame in frames:
        clean_text = frame["clean_title"].astype("string").fillna("").str.strip()
        title = frame["title"].astype("string").fillna("").str.strip()
        has_image = normalize_bool_series(frame["hasImage"])
        image_url = frame["image_url"].astype("string").fillna("").str.strip()
        created = pd.to_datetime(frame["created_utc"], unit="s", errors="coerce")
        row = {
            "subset": subset_name(is_mm),
            "split": split,
            "rows": len(frame),
            "columns": len(frame.columns),
            "unique_ids": int(frame["id"].nunique(dropna=True)),
            "duplicate_id_rows": int(frame["id"].astype("string").duplicated(keep=False).sum()),
            "missing_clean_title_rows": int(clean_text.eq("").sum()),
            "missing_title_rows": int(title.eq("").sum()),
            "has_image_rows": int(has_image.sum()),
            "image_url_rows": int(image_url.ne("").sum()),
            "distinct_subreddits": int(frame["subreddit"].nunique(dropna=True)),
            "distinct_domains": int(frame["domain"].nunique(dropna=True)),
            "distinct_authors": int(frame["author"].nunique(dropna=True)),
            "created_min": created.min(),
            "created_max": created.max(),
            "source_file": source_file,
        }
        row.update(quantile_summary(clean_text.str.len(), "clean_title_chars"))
        row.update(quantile_summary(clean_text.str.split().str.len(), "clean_title_words"))
        row.update(quantile_summary(frame["score"], "score"))
        row.update(quantile_summary(frame["num_comments"], "num_comments"))
        row.update(quantile_summary(frame["upvote_ratio"], "upvote_ratio"))
        rows.append(row)
    return pd.DataFrame(rows)


def build_label_distribution(frames: list[tuple[str, str, bool, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for _, split, is_mm, frame in frames:
        for column, names in LABEL_NAMES.items():
            counts = frame[column].value_counts(dropna=False).sort_index()
            total = len(frame)
            for value, count in counts.items():
                value_int = int(value) if pd.notna(value) else value
                rows.append(
                    {
                        "subset": subset_name(is_mm),
                        "split": split,
                        "label_column": column,
                        "label_value": value_int,
                        "label_name": names.get(value_int, str(value_int)),
                        "rows": int(count),
                        "share": safe_rate(int(count), total),
                    }
                )
    return pd.DataFrame(rows)


def top_values(
    frames: list[tuple[str, str, bool, pd.DataFrame]],
    columns: Iterable[str],
    top_n: int = 25,
) -> dict[str, pd.DataFrame]:
    outputs = {}
    for column in columns:
        rows = []
        for _, split, is_mm, frame in frames:
            counts = frame[column].astype("string").fillna("<missing>").value_counts(dropna=False).head(top_n)
            total = len(frame)
            for value, count in counts.items():
                rows.append(
                    {
                        "subset": subset_name(is_mm),
                        "split": split,
                        "column": column,
                        "value": value,
                        "rows": int(count),
                        "share": safe_rate(int(count), total),
                    }
                )
        outputs[column] = pd.DataFrame(rows)
    return outputs


def build_missingness(frames: list[tuple[str, str, bool, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for _, split, is_mm, frame in frames:
        for column in frame.columns:
            missing = int(frame[column].isna().sum())
            if frame[column].dtype == "object" or str(frame[column].dtype).startswith("string"):
                missing += int(frame[column].astype("string").fillna("").str.strip().eq("").sum())
            rows.append(
                {
                    "subset": subset_name(is_mm),
                    "split": split,
                    "column": column,
                    "missing_rows": missing,
                    "missing_share": safe_rate(missing, len(frame)),
                }
            )
    return pd.DataFrame(rows)


def duplicate_text_report(frames: list[tuple[str, str, bool, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for is_mm in [False, True]:
        subset_frames = []
        for _, split, frame_is_mm, frame in frames:
            if frame_is_mm != is_mm:
                continue
            temp = frame[["id", "clean_title"]].copy()
            temp["split"] = split
            temp["clean_title_norm"] = temp["clean_title"].astype("string").fillna("").str.strip().str.lower()
            temp = temp[temp["clean_title_norm"].ne("")]
            subset_frames.append(temp)
        combined = pd.concat(subset_frames, ignore_index=True)
        grouped = (
            combined.groupby("clean_title_norm")
            .agg(
                rows=("id", "size"),
                unique_ids=("id", "nunique"),
                splits=("split", lambda values: ",".join(sorted(set(values)))),
            )
            .reset_index()
        )
        grouped["split_count"] = grouped["splits"].str.count(",") + 1
        grouped = grouped[grouped["split_count"] > 1].sort_values(["rows", "unique_ids"], ascending=False)
        grouped.insert(0, "subset", subset_name(is_mm))
        rows.append(grouped.head(100))
    return pd.concat(rows, ignore_index=True)


def split_overlap_report(frames: list[tuple[str, str, bool, pd.DataFrame]], column: str) -> pd.DataFrame:
    rows = []
    for is_mm in [False, True]:
        parts = []
        for _, split, frame_is_mm, frame in frames:
            if frame_is_mm != is_mm or column not in frame.columns:
                continue
            values = frame[column].astype("string").fillna("").str.strip()
            part = pd.DataFrame({"split": split, "value": values})
            part = part[part["value"].ne("")]
            parts.append(part)
        combined = pd.concat(parts, ignore_index=True)
        grouped = (
            combined.groupby("value")
            .agg(rows=("split", "size"), splits=("split", lambda values: ",".join(sorted(set(values)))))
            .reset_index()
        )
        grouped["split_count"] = grouped["splits"].str.count(",") + 1
        grouped = grouped[grouped["split_count"] > 1].sort_values("rows", ascending=False)
        grouped.insert(0, "column", column)
        grouped.insert(0, "subset", subset_name(is_mm))
        rows.append(grouped.head(100))
    return pd.concat(rows, ignore_index=True)


def plot_label_distribution(label_distribution: pd.DataFrame, output_dir: Path) -> None:
    two_way = label_distribution[label_distribution["label_column"] == "2_way_label"].copy()
    two_way["share_pct"] = two_way["share"] * 100
    plt.figure(figsize=(10, 5))
    sns.barplot(data=two_way, x="split", y="share_pct", hue="label_name")
    plt.title("Fakeddit 2-way label share by split")
    plt.ylabel("Share (%)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "label_2_way_share.png", dpi=160)
    plt.close()


def plot_text_lengths(frames: list[tuple[str, str, bool, pd.DataFrame]], output_dir: Path) -> None:
    parts = []
    for _, split, is_mm, frame in frames:
        text_len = frame["clean_title"].astype("string").fillna("").str.strip().str.len()
        sample = text_len.sample(min(20000, len(text_len)), random_state=42)
        parts.append(pd.DataFrame({"subset": subset_name(is_mm), "split": split, "chars": sample}))
    text_lengths = pd.concat(parts, ignore_index=True)
    plt.figure(figsize=(11, 5))
    sns.histplot(data=text_lengths, x="chars", hue="subset", bins=60, stat="density", common_norm=False)
    plt.xlim(0, 220)
    plt.title("Clean title character length distribution")
    plt.xlabel("Characters in clean_title")
    plt.tight_layout()
    plt.savefig(output_dir / "clean_title_length_distribution.png", dpi=160)
    plt.close()


def write_markdown_report(
    output_path: Path,
    dataset_summary: pd.DataFrame,
    label_distribution: pd.DataFrame,
    missingness: pd.DataFrame,
    duplicate_text: pd.DataFrame,
    id_overlap: pd.DataFrame,
) -> None:
    mm_summary = dataset_summary[dataset_summary["subset"] == "multimodal_only"]
    all_summary = dataset_summary[dataset_summary["subset"] == "all_samples"]
    mm_train = mm_summary[mm_summary["split"] == "train"].iloc[0]
    all_train = all_summary[all_summary["split"] == "train"].iloc[0]

    missing_focus = missingness[
        missingness["column"].isin(["clean_title", "image_url", "linked_submission_id", "num_comments", "upvote_ratio"])
    ].sort_values(["subset", "split", "missing_share"], ascending=[True, True, False])

    two_way = label_distribution[label_distribution["label_column"] == "2_way_label"].copy()
    two_way["share_pct"] = (two_way["share"] * 100).round(2)

    lines = [
        "# Fakeddit EDA Report",
        "",
        "## Executive Summary",
        "",
        f"- Full training split contains {int(all_train['rows']):,} rows; multimodal-only training split contains {int(mm_train['rows']):,} rows.",
        "- The multimodal-only files have no missing `clean_title` values and every row has `hasImage=True`.",
        "- The full files include text-only/non-image rows and inherited `Unnamed` index columns, which should be dropped during modeling.",
        "- The existing local `private_image_set/` is not ID-aligned with these public TSVs. Use `data/raw/fakeddit/images/` for ID-named downloads from `image_url`.",
        "- Duplicate IDs are not present within the released splits, but repeated or generic `clean_title` values occur across splits and should be handled carefully for text-only experiments.",
        "",
        "## Row Counts",
        "",
        dataset_summary[
            [
                "subset",
                "split",
                "rows",
                "unique_ids",
                "missing_clean_title_rows",
                "has_image_rows",
                "image_url_rows",
                "distinct_subreddits",
                "distinct_domains",
            ]
        ].to_markdown(index=False),
        "",
        "## 2-way Label Balance",
        "",
        two_way[["subset", "split", "label_name", "rows", "share_pct"]].to_markdown(index=False),
        "",
        "## Text Length Summary",
        "",
        dataset_summary[
            [
                "subset",
                "split",
                "clean_title_chars_mean",
                "clean_title_chars_median",
                "clean_title_chars_p95",
                "clean_title_words_mean",
                "clean_title_words_median",
                "clean_title_words_p95",
            ]
        ].round(2).to_markdown(index=False),
        "",
        "## Missingness Focus",
        "",
        missing_focus.to_markdown(index=False),
        "",
        "## Leakage and Split Integrity",
        "",
        f"- Cross-split ID overlap rows found within comparable subsets: {len(id_overlap):,}.",
        f"- Top repeated clean-title rows written to `duplicate_clean_titles_across_splits.csv`; top report rows: {len(duplicate_text):,}.",
        "- Repeated `clean_title` values are often short/generic titles. This is not automatically fatal, but exact-text overlap should be considered when reporting text-only baselines.",
        "",
        "## Generated Figures",
        "",
        "- `figures/label_2_way_share.png`",
        "- `figures/clean_title_length_distribution.png`",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_fakeddit_config(args.config)
    output_dir = config.reports_dir / "deep_eda"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frames = read_split_frames(args.config)
    dataset_summary = build_dataset_summary(frames)
    label_distribution = build_label_distribution(frames)
    missingness = build_missingness(frames)
    duplicate_text = duplicate_text_report(frames)
    id_overlap = split_overlap_report(frames, "id")
    url_overlap = split_overlap_report(frames, "image_url")
    top_tables = top_values(frames, ["subreddit", "domain", "author"])

    dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    label_distribution.to_csv(output_dir / "label_distribution.csv", index=False)
    missingness.to_csv(output_dir / "missingness.csv", index=False)
    duplicate_text.to_csv(output_dir / "duplicate_clean_titles_across_splits.csv", index=False)
    id_overlap.to_csv(output_dir / "id_overlap_across_splits.csv", index=False)
    url_overlap.to_csv(output_dir / "image_url_overlap_across_splits.csv", index=False)
    for column, table in top_tables.items():
        table.to_csv(output_dir / f"top_{column}.csv", index=False)

    plot_label_distribution(label_distribution, figures_dir)
    plot_text_lengths(frames, figures_dir)
    write_markdown_report(
        output_dir / "eda_report.md",
        dataset_summary,
        label_distribution,
        missingness,
        duplicate_text,
        id_overlap,
    )
    print(f"Wrote detailed EDA report under {output_dir}")


if __name__ == "__main__":
    main()
