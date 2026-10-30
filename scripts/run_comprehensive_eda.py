"""Reproducible, bounded EDA over the local verified FNEWS artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/research_dataset"
OUT = ROOT / "reports/EDA"
FIG = OUT / "figures"
TAB = OUT / "tables"
SEED = 42


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=180, bbox_inches="tight")
    plt.close()


def paired(split: str) -> pd.DataFrame:
    f = pd.read_parquet(DATA / "manifests" / f"{split}_manifest.parquet")
    return f[
        (f.source_dataset == "Fakeddit") & (f.download_status == "verified_reused_or_processed")
    ].copy()


def label(raw: pd.Series) -> pd.Series:
    return np.where(raw.astype(int).eq(1), "SOURCE_TRUE", "SOURCE_NON_TRUE")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    pieces = []
    inventory = []
    for split in ("train", "validate", "test"):
        raw = pd.read_parquet(DATA / "manifests" / f"{split}_manifest.parquet")
        valid = paired(split).reset_index(drop=True)
        valid["split"] = split
        valid["source_label"] = label(valid.fake_news_label)
        pieces.append(valid)
        inventory.append(
            {
                "dataset": "Fakeddit research-v1",
                "split": split,
                "manifest_rows": len(raw),
                "verified_pairs": len(valid),
                "text_available": int(valid.text.notna().sum()),
                "image_available": int(valid.image_path.notna().sum()),
                "raw_label_1_source_true": int((valid.fake_news_label == 1).sum()),
                "raw_label_0_source_non_true": int((valid.fake_news_label == 0).sum()),
            }
        )
    df = pd.concat(pieces, ignore_index=True)
    inv = pd.DataFrame(inventory)
    inv.to_csv(OUT / "dataset_inventory.csv", index=False)
    df["chars"] = df.text.str.len()
    df["words"] = df.text.str.split().str.len()
    df["sentence_count"] = df.text.str.count(r"[.!?]") + 1
    df["punctuation"] = df.text.str.count(r"[^\w\s]")
    df["uppercase_ratio"] = df.text.str.count(r"[A-Z]") / df.chars.clip(lower=1)
    df["question_marks"] = df.text.str.count(r"\?")
    df["exclamation_marks"] = df.text.str.count("!")
    # Quality summary is measured on the processed, actually loadable subset.
    q = []
    for split, group in df.groupby("split"):
        q.append(
            {
                "split": split,
                "rows": len(group),
                "missing_text": int(group.text.isna().sum()),
                "empty_text": int(group.text.fillna("").str.strip().eq("").sum()),
                "missing_image_path": int(group.image_path.isna().sum()),
                "duplicate_text_members": int(group.duplicated("text", keep=False).sum()),
                "duplicate_image_path_members": int(
                    group.duplicated("image_path", keep=False).sum()
                ),
                "invalid_label": int((~group.fake_news_label.isin([0, 1])).sum()),
            }
        )
    quality = pd.DataFrame(q)
    quality.to_csv(OUT / "data_quality_summary.csv", index=False)
    # Image inspection: all local verified files; failures remain in the table.
    rows = []
    for _i, row in df.iterrows():
        path = DATA / row.image_path
        record = {
            "sample_id": row.sample_id,
            "split": row.split,
            "source_label": row.source_label,
            "path": row.image_path,
            "readable": False,
        }
        try:
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                a = np.asarray(rgb.resize((64, 64)), dtype=float)
                record.update(
                    {
                        "readable": True,
                        "width": im.width,
                        "height": im.height,
                        "format": im.format,
                        "mode": im.mode,
                        "bytes": path.stat().st_size,
                        "brightness": float(a.mean()),
                        "contrast": float(a.std()),
                        "aspect": im.width / im.height,
                    }
                )
        except Exception as exc:
            record["error"] = type(exc).__name__
        rows.append(record)
    images = pd.DataFrame(rows)
    images.to_csv(TAB / "image_features.csv", index=False)
    # Join six-way taxonomy from official local metadata only.
    meta = []
    for p in (ROOT / "data/raw/fakeddit/multimodal_only_samples").glob("*.tsv"):
        split = "test" if "test" in p.name else "validate" if "validate" in p.name else "train"
        x = pd.read_csv(
            p,
            sep="\t",
            usecols=[
                "id",
                "clean_title",
                "subreddit",
                "domain",
                "2_way_label",
                "3_way_label",
                "6_way_label",
            ],
        )
        x["original_id"] = x.id.astype(str)
        x["text"] = x.clean_title.astype(str).str.strip()
        x["split"] = split
        meta.append(x)
    metadata = pd.concat(meta, ignore_index=True)
    df = df.merge(metadata, on=["original_id", "text", "split"], how="left", validate="many_to_one")
    df.to_parquet(TAB / "paired_records_with_metadata.parquet", index=False)
    # Leakage: exact text/hash groups spanning a split are confirmed; phash across split is possible near duplicate evidence.
    leakage = []
    for field, kind in [
        ("text", "exact_text"),
        ("image_hash", "exact_image_hash"),
        ("perceptual_hash", "perceptual_hash"),
    ]:
        tmp = df.groupby(field).agg(
            splits=("split", lambda x: sorted(set(x))), members=("sample_id", "size")
        )
        cross = tmp[tmp.splits.map(len) > 1]
        leakage.append(
            {
                "check": kind,
                "cross_split_groups": len(cross),
                "cross_split_members": int(cross.members.sum()),
                "classification": "confirmed" if kind != "perceptual_hash" else "possible",
            }
        )
    pd.DataFrame(leakage).to_csv(OUT / "leakage_audit.csv", index=False)
    # Statistical comparisons, two-sided MW with rank-biserial effect size.
    tests = []
    for feature, table in [
        ("chars", df),
        ("words", df),
        ("brightness", images[images.readable]),
        ("contrast", images[images.readable]),
        ("aspect", images[images.readable]),
    ]:
        a = table.loc[table.source_label == "SOURCE_TRUE", feature].dropna()
        b = table.loc[table.source_label == "SOURCE_NON_TRUE", feature].dropna()
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        rbc = 1 - (2 * u) / (len(a) * len(b))
        tests.append(
            {
                "feature": feature,
                "n_true": len(a),
                "n_non_true": len(b),
                "median_true": float(a.median()),
                "median_non_true": float(b.median()),
                "mann_whitney_p": p,
                "rank_biserial_effect": rbc,
            }
        )
    pd.DataFrame(tests).to_csv(TAB / "statistical_tests.csv", index=False)
    # Text terms with exploratory tokenization; this is not causal evidence.
    vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 3), min_df=3, stop_words="english")
    matrix = vectorizer.fit_transform(df.text)
    terms = np.asarray(vectorizer.get_feature_names_out())
    term_rows = []
    for cls in ["SOURCE_TRUE", "SOURCE_NON_TRUE"]:
        mean = np.asarray(matrix[df.source_label.eq(cls).to_numpy()].mean(axis=0)).ravel()
        idx = mean.argsort()[-20:][::-1]
        term_rows.extend(
            {"source_label": cls, "term": terms[j], "mean_count": mean[j]} for j in idx
        )
    pd.DataFrame(term_rows).to_csv(TAB / "top_ngrams.csv", index=False)
    # Figures: every chart is artifact-backed and named for its question.
    colors = {"SOURCE_TRUE": "#4e8b6a", "SOURCE_NON_TRUE": "#b86753"}
    inv.set_index("split")[["verified_pairs"]].plot.bar(
        title="Verified paired records by split", legend=False, color="#4e8b6a"
    )
    plt.ylabel("pairs")
    savefig("01_split_pairs.png")
    inv.set_index("split")[["raw_label_1_source_true", "raw_label_0_source_non_true"]].plot.bar(
        stacked=True,
        title="Source taxonomy by split",
        color=[colors["SOURCE_TRUE"], colors["SOURCE_NON_TRUE"]],
    )
    plt.ylabel("pairs")
    savefig("02_class_by_split.png")
    quality.set_index("split")[
        ["missing_text", "empty_text", "missing_image_path", "invalid_label"]
    ].plot.bar(title="Processed-pair missingness and invalid labels")
    plt.ylabel("records")
    savefig("03_quality_missingness.png")
    for feature, num, title in [
        ("chars", 4, "Character count by source taxonomy"),
        ("words", 5, "Word count by source taxonomy"),
        ("sentence_count", 6, "Sentence-count proxy by source taxonomy"),
        ("punctuation", 7, "Punctuation count by source taxonomy"),
    ]:
        for cls, g in df.groupby("source_label"):
            plt.hist(g[feature], bins=40, alpha=0.55, label=cls, color=colors[cls])
        plt.title(title)
        plt.xlabel(feature)
        plt.ylabel("records")
        plt.legend()
        savefig(f"{num:02d}_{feature}.png")
    df.boxplot(column="chars", by="source_label")
    plt.suptitle("")
    plt.title("Character count distribution by source taxonomy")
    plt.ylabel("characters")
    savefig("08_chars_boxplot.png")
    df.boxplot(column="uppercase_ratio", by="source_label")
    plt.suptitle("")
    plt.title("Capital-letter ratio by source taxonomy")
    plt.ylabel("ratio")
    savefig("09_uppercase_ratio.png")
    top = pd.read_csv(TAB / "top_ngrams.csv")
    for cls, num in [("SOURCE_TRUE", 10), ("SOURCE_NON_TRUE", 11)]:
        s = top[top.source_label == cls].sort_values("mean_count").tail(15)
        plt.barh(s.term, s.mean_count, color=colors[cls])
        plt.title(f"Exploratory frequent n-grams: {cls}")
        plt.xlabel("mean document count")
        savefig(f"{num:02d}_ngrams_{cls.lower()}.png")
    clean = images[images.readable]
    plt.scatter(clean.width, clean.height, c=clean.source_label.map(colors), alpha=0.45, s=9)
    plt.title("Image resolution by source taxonomy")
    plt.xlabel("width (px)")
    plt.ylabel("height (px)")
    savefig("12_resolution_scatter.png")
    for feature, num, title in [
        ("aspect", 13, "Image aspect ratio"),
        ("brightness", 14, "Image brightness"),
        ("contrast", 15, "Image contrast"),
        ("bytes", 16, "Image file size"),
    ]:
        for cls, g in clean.groupby("source_label"):
            plt.hist(g[feature], bins=35, alpha=0.55, label=cls, color=colors[cls])
        plt.title(title + " by source taxonomy")
        plt.xlabel(feature)
        plt.ylabel("images")
        plt.legend()
        savefig(f"{num:02d}_{feature}.png")
    clean["format"].value_counts().plot.bar(title="Verified image formats", color="#476f99")
    plt.ylabel("images")
    savefig("17_image_formats.png")
    pd.DataFrame(leakage).set_index("check")["cross_split_groups"].plot.bar(
        title="Cross-split duplicate-risk groups", color="#7c6aa6"
    )
    plt.ylabel("groups")
    savefig("18_leakage_groups.png")
    # Fakeddit's official field is literally named ``6_way_label``.  Keep that
    # name rather than inventing a renamed taxonomy column.
    six_way = next(
        column
        for column in ("6_way_label", "6_way_label_y", "6_way_label_x")
        if column in df.columns
    )
    df.loc[df.source_label == "SOURCE_NON_TRUE", six_way].value_counts().sort_index().plot.bar(
        title="SOURCE_NON_TRUE six-way subtype distribution", color="#b86753"
    )
    plt.xlabel("six-way subtype")
    plt.ylabel("records")
    savefig("19_non_true_subtypes.png")
    # contact sheet from reproducible class-balanced sample
    sample = df.groupby("source_label", group_keys=False).sample(6, random_state=SEED)
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    for ax, (_, r) in zip(axes.flat, sample.iterrows(), strict=True):
        ax.imshow(Image.open(DATA / r.image_path).convert("RGB"))
        ax.set_title(r.source_label + "\n" + r.text[:30])
        ax.axis("off")
    plt.suptitle("Class-balanced verified image contact sheet", y=1.02)
    savefig("20_contact_sheet.png")
    summary = {
        "verified_pairs": len(df),
        "splits": inv.to_dict("records"),
        "readable_images": int(clean.shape[0]),
        "corrupt_images": int((~images.readable).sum()),
        "exact_text_cross_split_groups": int(
            pd.read_csv(OUT / "leakage_audit.csv")
            .query("check == 'exact_text'")
            .cross_split_groups.iloc[0]
        ),
        "exact_image_cross_split_groups": int(
            pd.read_csv(OUT / "leakage_audit.csv")
            .query("check == 'exact_image_hash'")
            .cross_split_groups.iloc[0]
        ),
    }
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
