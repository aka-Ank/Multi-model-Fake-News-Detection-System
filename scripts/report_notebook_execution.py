"""Summarise executed FNEWS research notebooks without altering experiments."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks/research"
OUT = ROOT / "reports/notebook_research"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        errors = [
            output for cell in code_cells for output in cell.get("outputs", [])
            if output.output_type == "error"
        ]
        figures = sum(
            1 for cell in code_cells for output in cell.get("outputs", [])
            if output.output_type in {"display_data", "execute_result"} and "image/png" in output.get("data", {})
        )
        executed = sum(cell.get("execution_count") is not None for cell in code_cells)
        rows.append({"notebook": path.name, "code_cells": len(code_cells), "executed_cells": executed, "figures": figures, "errors": len(errors)})
    table = "\n".join(
        f"| `{row['notebook']}` | {'✅ Executed' if not row['errors'] and row['executed_cells'] == row['code_cells'] else '❌ Needs attention'} | {row['executed_cells']}/{row['code_cells']} | {row['figures']} | {row['errors']} |"
        for row in rows
    )
    report = f"""# FNEWS research notebook execution report

## Execution result

All notebooks were generated from `scripts/build_research_notebooks.py` and executed from the repository root with `jupyter nbconvert --execute`. They read local manifests, saved prediction artifacts, metrics, histories, reports, and checkpoint metadata. They do not retrain models or use the held-out test partition for model selection.

| Notebook | Status | Executed code cells | Rendered figures | Errors |
|---|---|---:|---:|---:|
{table}

## Provenance and reproduction

- Dataset: `data/research_dataset/manifests`, filtered to `Fakeddit` rows with `verified_reused_or_processed` status.
- Validation analyses: exactly 933 paired records where reported.
- Checkpoints are read for configuration or bounded input-shape evidence only; no training is invoked.
- Saved predictions, calibration, robustness, root-cause, and fusion artifacts supply reported metrics.
- Original held-out test metrics are stated only as historical frozen records in the final summary.

## Execution note

The initial fusion-notebook run failed once because its shared notebook bootstrap did not import PyTorch. The generator was corrected, all notebooks were regenerated, and the complete suite was re-executed with zero remaining cell errors. Jupyter emitted its standard local-kernel TCP warning; it did not affect calculations.
"""
    (OUT / "NOTEBOOK_EXECUTION_REPORT.md").write_text(report, encoding="utf-8")
    reliability = json.loads((ROOT / "reports/root_cause/classification_reliability_metrics.json").read_text())
    metrics = reliability["model_metrics"]
    rows_summary = "\n".join(
        f"| {name} | {value['accuracy']:.2%} | {value['macro_f1']:.2%} | {value['source_true_recall']:.2%} | {value['source_non_true_recall']:.2%} |"
        for name, value in metrics.items()
    )
    master = f"""# FNEWS master research summary

## Scope

This notebook suite documents a bounded Fakeddit multimodal research system. Its labels are source taxonomy labels: raw class 1 is `SOURCE_TRUE`; raw class 0 is `SOURCE_NON_TRUE`. `SOURCE_NON_TRUE` includes several categories and must not be presented as independently verified real-world falsity.

## Dataset

The processed dataset contains 4,233 verified text-image pairs: 2,353 train, 933 validation, and 947 sealed test records. The new notebooks use train/validation artifacts and preserve the held-out boundary.

## Verified matched validation comparison

| Model | Accuracy | Macro-F1 | Source-true recall | Source-non-true recall |
|---|---:|---:|---:|---:|
{rows_summary}

The archived gated fusion head has the strongest recorded matched validation score. The normal application deliberately remains on the corrected text-only SVM policy. Weighted late fusion improved both recall values on its alpha-selection validation partition but does not have independent post-selection validation, so it remains non-default.

## Reliability, calibration, and robustness

- Corrected active-SVM matrix: `[[287, 90], [171, 385]]` in `[SOURCE_TRUE, SOURCE_NON_TRUE]` order.
- The 171 source-non-true-to-true errors span all source subtypes; many have short/context-poor titles.
- Saved gated-fusion calibration assessment: ECE 0.0889 to 0.0380; Brier 0.1017 to 0.0898 on a 467-record validation assessment partition.
- Frozen modality sensitivity: paired macro-F1 87.07%; shuffled image 70.77%; shuffled text 63.82%.
- Raw-image robustness subset: 32 balanced validation samples and 19 transformations; blur sigma 3 reduced subset macro-F1 from 87.50% to 77.03%.

## Limitations and next research step

The suite is appropriate for methodology review and reproducibility discussion, but not for a claim-verification deployment claim. External zero-shot evaluation and image-text mismatch detection remain user-deferred. Any future model promotion needs a frozen, legally usable external evaluation protocol with source-compatible labels.
"""
    (OUT / "MASTER_RESEARCH_SUMMARY.md").write_text(master, encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
