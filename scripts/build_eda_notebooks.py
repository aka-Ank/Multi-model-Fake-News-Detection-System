"""Build executed-notebook-ready, artifact-backed FNEWS EDA notebooks."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "EDA"
BOOT = """from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nROOT = Path.cwd()\nwhile not (ROOT / 'reports' / 'EDA').exists(): ROOT = ROOT.parent\nEDA = ROOT / 'reports' / 'EDA'\nFIG = EDA / 'figures'\nsummary = json.loads((EDA / 'eda_summary.json').read_text())\nsummary"""

SPECS = [
    (
        "01_dataset_inventory",
        "Dataset Inventory and Provenance",
        "Inventories only verified local Fakeddit image–text pairs. The raw taxonomy is SOURCE_TRUE (1) and SOURCE_NON_TRUE (0), not an independently fact-checked verdict.",
        "inventory=pd.read_csv(EDA/'dataset_inventory.csv')\ndisplay(inventory)\nfor n in ['01_split_pairs.png','02_class_by_split.png']: display(Image(filename=str(FIG/n)))",
    ),
    (
        "02_data_quality",
        "Data Quality and Missingness",
        "Checks text/image presence, labels, duplicate paths, and decodability of processed local images. It does not claim coverage of complete upstream releases.",
        "quality=pd.read_csv(EDA/'data_quality_summary.csv'); images=pd.read_csv(EDA/'tables'/'image_features.csv')\ndisplay(quality); display(images.readable.value_counts(dropna=False).rename('count').to_frame())\nfor n in ['03_quality_missingness.png','17_image_formats.png']: display(Image(filename=str(FIG/n)))",
    ),
    (
        "03_label_analysis",
        "Label Semantics and Class Balance",
        "Source taxonomy labels are not fact-checked factuality verdicts. NewsCLIPpings mismatch labels are excluded because there are zero validated local paired mismatch images.",
        "records=pd.read_parquet(EDA/'tables'/'paired_records_with_metadata.parquet')\ndisplay(records.groupby(['split','source_label']).size().unstack(fill_value=0))\ndisplay(records.loc[records.source_label.eq('SOURCE_NON_TRUE'),'6_way_label'].value_counts(dropna=False).rename('records').to_frame())\nfor n in ['02_class_by_split.png','19_non_true_subtypes.png']: display(Image(filename=str(FIG/n)))",
    ),
    (
        "04_text_eda",
        "Text EDA",
        "Text measures and n-grams are exploratory shortcut-risk diagnostics, not causal explanations or article-level truth claims.",
        "stats=pd.read_csv(EDA/'tables'/'statistical_tests.csv'); top=pd.read_csv(EDA/'tables'/'top_ngrams.csv')\ndisplay(stats[stats.feature.isin(['chars','words'])]); display(top.groupby('source_label').head(10))\nfor n in ['04_chars.png','05_words.png','08_chars_boxplot.png','09_uppercase_ratio.png','10_ngrams_source_true.png','11_ngrams_source_non_true.png']: display(Image(filename=str(FIG/n)))",
    ),
    (
        "05_image_eda",
        "Image EDA",
        "Image measurements come from exact processed bytes. Pixel distributions are potential artifact signals, not semantic image understanding.",
        "images=pd.read_csv(EDA/'tables'/'image_features.csv')\ndisplay(images.groupby('source_label')[['width','height','brightness','contrast','aspect','bytes']].describe().round(2))\nfor n in ['12_resolution_scatter.png','13_aspect.png','14_brightness.png','15_contrast.png','16_bytes.png','20_contact_sheet.png']: display(Image(filename=str(FIG/n)))",
    ),
    (
        "06_multimodal_eda",
        "Multimodal Pairing and Coverage",
        "Every included record has non-empty text and a readable local image. Pairings are taken from processed manifests.",
        "records=pd.read_parquet(EDA/'tables'/'paired_records_with_metadata.parquet')\ndisplay(records.groupby('split').agg(pairs=('sample_id','size'),text_nonempty=('text',lambda s:s.fillna('').str.strip().ne('').sum()),image_hashes=('image_hash','nunique')))\nfor n in ['01_split_pairs.png','20_contact_sheet.png']: display(Image(filename=str(FIG/n)))",
    ),
    (
        "07_leakage_audit",
        "Split Leakage and Duplicate Audit",
        "Exact duplicate text across splits is confirmed leakage risk. Exact image hashes do not cross splits. Perceptual hash groups are candidate evidence, not a complete visual near-duplicate proof.",
        "leakage=pd.read_csv(EDA/'leakage_audit.csv'); display(leakage); display(Image(filename=str(FIG/'18_leakage_groups.png')) )",
    ),
    (
        "08_statistical_analysis",
        "Statistical Analysis",
        "Two-sided Mann–Whitney U tests compare taxonomy distributions. Rank-biserial correlation communicates effect size; significance alone is not practical importance.",
        "tests=pd.read_csv(EDA/'tables'/'statistical_tests.csv'); display(tests.sort_values('mann_whitney_p')); display(Markdown('Scope: diagnostics for the curated local subset, not causal or external-generalization evidence.'))",
    ),
    (
        "09_model_implications",
        "Model Implications and Responsible Next Steps",
        "EDA informs safeguards, not model selection. The sealed test split is not used for model evaluation or tuning here.",
        "tests=pd.read_csv(EDA/'tables'/'statistical_tests.csv'); leakage=pd.read_csv(EDA/'leakage_audit.csv')\ndisplay(Markdown(f'''### Evidence-backed implications\n- Text-length shortcut risk: medians are {tests.loc[tests.feature=='chars','median_true'].iloc[0]:.0f} versus {tests.loc[tests.feature=='chars','median_non_true'].iloc[0]:.0f} characters.\n- Leakage control: {int(leakage.loc[leakage.check=='exact_text','cross_split_groups'].iloc[0])} exact-text groups cross splits.\n- Image integrity: {summary['readable_images']}/{summary['verified_pairs']} images decoded.\n- Source taxonomy is not factuality verification.'''))\nfor n in ['08_chars_boxplot.png','18_leakage_groups.png']: display(Image(filename=str(FIG/n)))",
    ),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for slug, title, intro, code in SPECS:
        nb = nbf.v4.new_notebook()
        nb.metadata.kernelspec = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        nb.cells = [
            nbf.v4.new_markdown_cell(f"# FNEWS EDA — {title}\n\n{intro}"),
            nbf.v4.new_code_cell(BOOT),
            nbf.v4.new_code_cell(code),
            nbf.v4.new_markdown_cell(
                "### Reproducibility note\n\nGenerated from local manifests and processed images by `scripts/run_comprehensive_eda.py`. No external content was fetched and no model training occurred."
            ),
        ]
        nbf.write(nb, OUT / f"{slug}.ipynb")


if __name__ == "__main__":
    main()
