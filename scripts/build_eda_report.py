"""Render the FNEWS research-grade EDA report from measured artifact tables."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EDA = ROOT / "reports" / "EDA"


def main() -> None:
    summary = json.loads((EDA / "eda_summary.json").read_text())
    inventory = pd.read_csv(EDA / "dataset_inventory.csv")
    quality = pd.read_csv(EDA / "data_quality_summary.csv")
    leakage = pd.read_csv(EDA / "leakage_audit.csv")
    stats = pd.read_csv(EDA / "tables" / "statistical_tests.csv")
    pairs = pd.read_parquet(EDA / "tables" / "paired_records_with_metadata.parquet")
    chars = stats.set_index("feature").loc["chars"]
    words = stats.set_index("feature").loc["words"]
    text_leak = leakage.set_index("check").loc["exact_text"]
    inv_md = inventory.to_markdown(index=False)
    quality_md = quality.to_markdown(index=False)
    leakage_md = leakage.to_markdown(index=False)
    stats_md = stats.to_markdown(index=False, floatfmt=".4g")
    subtype = (
        pairs.loc[pairs.source_label.eq("SOURCE_NON_TRUE"), "6_way_label"]
        .value_counts(dropna=False)
        .to_frame("records")
        .to_markdown()
    )
    report = f"""# FNEWS Comprehensive Research-Grade EDA

## Abstract

This report audits the **locally verified Fakeddit research-v1 subset** used by FNEWS. It covers {summary["verified_pairs"]:,} decodable image–text pairs, not the complete upstream Fakeddit release. The analysis is descriptive and reproducible: it does not train models, tune a threshold, or use held-out test labels for model selection. The dataset labels are **source taxonomy labels**, not independently fact-checked truth judgments.

## Data Inventory and Provenance

The source manifests preserve original split membership. Only records with `source_dataset=Fakeddit` and `download_status=verified_reused_or_processed` enter analysis. Images are processed local files, with provenance retained in the manifests and paired-record table.

{inv_md}

**Observation → Evidence → implication.** The working corpus is compact and fully paired, while each raw manifest is much larger. Therefore dataset-wide conclusions apply to this controlled research subset only. The local NewsCLIPpings pairing task remains unavailable (zero validated image–text pairs) and is deliberately not folded into Fakeddit labels.

## Label Semantics and Class Balance

Fakeddit raw two-way label `1` is rendered as `SOURCE_TRUE`; raw `0` is `SOURCE_NON_TRUE`. `SOURCE_NON_TRUE` is a dataset-defined umbrella category and should not be described as verified fake news. The subtype distribution among local `SOURCE_NON_TRUE` examples is:

{subtype}

**Recommendation.** Preserve raw and normalized source labels side by side in all future exports and UI responses. Never claim website factual verification from these labels.

## Data Quality and Modality Coverage

{quality_md}

All {summary["readable_images"]:,}/{summary["verified_pairs"]:,} local images decoded successfully; the measured corrupt-image count is {summary["corrupt_images"]}. Text and image paths are present for every included pair. This removes corrupt processed bytes as an explanation for current baseline behavior, but does not prove semantic relevance or correctness of every image.

## Text Analysis

Median text length differs materially by source taxonomy: `SOURCE_TRUE` has {chars.median_true:.0f} characters and {words.median_true:.0f} words versus {chars.median_non_true:.0f} characters and {words.median_non_true:.0f} words for `SOURCE_NON_TRUE`. The two-sided Mann–Whitney results are shown below. This is a potential text-length shortcut risk, not evidence that length determines factuality.

## Image Analysis

Image resolution, aspect ratio, brightness, contrast, bytes, and formats were measured from decoded image bytes. Brightness, contrast, and aspect-ratio effects are small in the measured comparison; image-level semantic relevance still requires human or task-specific validation. Use augmentation and controlled modality ablations before treating image features as causal evidence.

## Multimodal Coverage and Pairing

Every analyzed record has both text and a readable image. The deterministic class-balanced contact sheet (`figures/20_contact_sheet.png`) supports visual spot-checking only. It is not a dataset representativeness claim. Valid Fakeddit pair identity is tracked by `sample_id`, `original_id`, image hash, and manifest split.

## Split Leakage Audit

{leakage_md}

**Confirmed risk:** {int(text_leak.cross_split_groups)} exact-text groups ({int(text_leak.cross_split_members)} records) occur across splits. **Positive integrity result:** no exact image-hash groups cross splits. The perceptual-hash check is conservative candidate evidence, not a complete Hamming-distance near-duplicate audit.

**Recommendation.** For any future retraining or new split construction, group/deduplicate identical text (and, where possible, source/post lineage) before allocation. Do not erase the existing frozen split or reinterpret historical test results.

## Statistical Analysis

{stats_md}

The Mann–Whitney tests are two-sided and rank-biserial effect sizes are included. P-values are descriptive for this curated subset; practical effect size, confounding, and split leakage matter more than a small p-value.

## Model Implications

1. **Evaluate against shortcut risk.** Text-length differences are substantial, so report length-controlled validation slices and compare against simple length-only baselines before attributing performance to factual reasoning.
2. **Protect split integrity.** Exact cross-split text duplicates can inflate text-model estimates. Keep the sealed test boundary intact, and use a new grouped protocol for any future research comparison.
3. **Keep modality claims narrow.** Clean image decoding enables image experiments but does not prove that image content is semantically paired or causal.
4. **Use calibrated language.** Scores derived from Fakeddit taxonomy do not verify a live claim, publisher, or article.
5. **Track missing modality separately.** A future acquisition with absent/misaligned images must fall back explicitly rather than being reported as multimodal.

## Recommended Next Steps

1. Freeze this EDA version with its manifest hashes and figures.
2. Add a grouped, text-deduplicated *future* validation protocol without changing frozen baseline results.
3. Run pre-registered, validation-only length/source controlled slices for active models.
4. Use visual QA samples to review article-image semantic alignment; label the result as manual audit.
5. Maintain NewsCLIPpings mismatch work as deferred until valid paired images exist.

## Artifacts

- Nine executed notebooks: `notebooks/EDA/01_dataset_inventory.ipynb` through `09_model_implications.ipynb`
- Twenty data-derived figures: `reports/EDA/figures/`
- Tables: `reports/EDA/dataset_inventory.csv`, `data_quality_summary.csv`, `leakage_audit.csv`, and `tables/`
- Reproduction scripts: `scripts/run_comprehensive_eda.py`, `scripts/build_eda_notebooks.py`, `scripts/build_eda_report.py`

## Limitations

- Findings characterize the verified local research subset, not every Fakeddit record or live web news.
- Source taxonomy labels are not fact-checked ground truth.
- The analysis does not create a new leakage-free split and does not revise previous frozen test claims.
- Image-level statistics do not establish image-caption relevance or explain individual predictions.
"""
    (EDA / "EDA_RESEARCH_REPORT.md").write_text(report, encoding="utf-8")
    (EDA / "recommendations.md").write_text(
        "# FNEWS EDA Recommendations\n\nSee `EDA_RESEARCH_REPORT.md` sections **Model Implications** and **Recommended Next Steps**. The priority is a future grouped/deduplicated validation protocol; preserve the frozen historical split and checkpoint.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
