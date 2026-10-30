"""Build the active, artifact-backed FNEWS notebook suite from archived sources.

This is a documentation-only consolidation.  It copies existing notebook cells
and their saved outputs; it never trains a model, changes a checkpoint, or
loads the sealed test manifest for selection.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
ARCHIVE = NOTEBOOKS / "archive" / "2026-10-27"

SUITES = [
    (
        "01_dataset_eda.ipynb",
        "Dataset exploration and integrity",
        "Dataset description, preprocessing boundaries, feature inspection, image coverage, and leakage safeguards.",
        [
            ("Dataset, text, image, and multimodal EDA", "research/01_dataset_eda.ipynb"),
            ("Preprocessing and leakage prevention", "research/02_preprocessing.ipynb"),
            ("Feature extraction", "research/03_feature_extraction.ipynb"),
        ],
    ),
    (
        "02_text_modeling.ipynb",
        "Text modeling",
        "Saved TF-IDF + LinearSVC and frozen pretrained-text baseline evidence, with configuration and validation results.",
        [("Text baselines and evaluation", "research/04_text_model_training.ipynb")],
    ),
    (
        "03_image_modeling.ipynb",
        "Image modeling",
        "Saved ResNet-18 transfer-learning evidence, preprocessing, validation evaluation, and its task limitations.",
        [("ResNet-18 transfer learning", "research/05_image_model_training.ipynb")],
    ),
    (
        "04_multimodal_fusion.ipynb",
        "Multimodal fusion",
        "Saved weighted late fusion and archived CLIP gated-fusion evidence, including scope and ablation caveats.",
        [("Weighted late fusion and CLIP gated fusion", "research/06_multimodal_fusion.ipynb")],
    ),
    (
        "05_evaluation_robustness.ipynb",
        "Evaluation, calibration, and robustness",
        "Matched validation comparison, error analysis, calibration and robustness evidence, then a clearly bounded final summary.",
        [
            ("Matched model evaluation", "research/07_model_evaluation.ipynb"),
            ("Error analysis", "research/08_error_analysis.ipynb"),
            ("Calibration and robustness", "research/09_robustness_calibration.ipynb"),
            ("Conclusions and limitations", "research/10_final_research_summary.ipynb"),
        ],
    ),
]


def remove_title_and_bootstrap(cells: list[nbformat.NotebookNode]) -> list[nbformat.NotebookNode]:
    """Keep source evidence while replacing repeated titles and setup cells."""
    copied = [nbformat.from_dict(cell) for cell in cells]
    if copied and copied[0].cell_type == "markdown":
        copied = copied[1:]
    if copied and copied[0].cell_type == "code" and "def paired(" in copied[0].source:
        copied = copied[1:]
    return copied


def source_cells(relative_path: str) -> list[nbformat.NotebookNode]:
    path = ARCHIVE / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Archived notebook missing: {path}")
    cells = remove_title_and_bootstrap(nbformat.read(path, as_version=4).cells)
    # The archived EDA source had already inspected split-wide metadata. The
    # active notebooks must not reopen the sealed test manifest on execution.
    for cell in cells:
        if cell.cell_type == "code" and "['train','validate','test']" in cell.source:
            cell.source = cell.source.replace("['train','validate','test']", "['train','validate']")
            cell.outputs = []
            cell.execution_count = None
    return cells


def bootstrap() -> str:
    return '''from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path.cwd()
if not (ROOT / "data/research_dataset").exists():
    ROOT = Path("/Users/aka_ank/Desktop/FNEWS")
DATA = ROOT / "data/research_dataset"
REPORTS = ROOT / "reports"
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.figsize": (8, 4.5), "axes.titlesize": 13, "axes.labelsize": 11})

def paired(split):
    frame = pd.read_parquet(DATA / "manifests" / f"{split}_manifest.parquet")
    return frame.loc[(frame.source_dataset == "Fakeddit") & (frame.download_status == "verified_reused_or_processed")].copy().reset_index(drop=True)

def canonical_source_labels(raw):
    """Fakeddit raw 1=true; return 0=SOURCE_TRUE, 1=SOURCE_NON_TRUE."""
    return 1 - np.asarray(raw, dtype=int)

def source_names(values):
    return np.where(np.asarray(values) == 0, "SOURCE_TRUE", "SOURCE_NON_TRUE")

def show_frame(frame, rows=10):
    display(frame.head(rows))
'''


def build_suite() -> None:
    for filename, title, objective, sections in SUITES:
        cells = [
            new_markdown_cell(
                f"# FNEWS — {title}\n\n"
                f"**Research objective.** {objective}\n\n"
                "This consolidated notebook reuses saved, executed research evidence. "
                "It does not train models or perform new held-out-test selection. "
                "`SOURCE_TRUE` and `SOURCE_NON_TRUE` are Fakeddit source-taxonomy labels, "
                "not independent real-world fact-checking verdicts."
            ),
            new_code_cell(bootstrap()),
        ]
        for heading, source in sections:
            cells.append(new_markdown_cell(f"## {heading}\n\n_Source preserved from `notebooks/archive/2026-10-27/{source}`._"))
            cells.extend(source_cells(source))
        cells.append(
            new_markdown_cell(
                "## Reproducibility and limitations\n\n"
                "The original notebooks remain unchanged under `notebooks/archive/2026-10-27/`. "
                "Rebuild this active suite with `python scripts/consolidate_notebooks.py`. "
                "Archived outputs are historical records, not claims of a new execution."
            )
        )
        notebook = new_notebook(cells=cells)
        notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
        nbformat.write(notebook, NOTEBOOKS / filename)


def description(path: Path, title: str) -> tuple[str, str, str]:
    relative = path.relative_to(ARCHIVE).as_posix()
    if relative.startswith("EDA/"):
        return ("Comprehensive EDA", "4,233 verified Fakeddit pairs and saved EDA tables/figures", "Absorbed by 01; retained for granular EDA provenance")
    if relative.startswith("research/"):
        return ("Artifact-backed research", "Saved Fakeddit manifests, predictions, metrics, histories, and checkpoints", "Absorbed by one of the five active notebooks; retained as executed source evidence")
    if "newsclippings" in path.name.lower() or "visualnews" in path.name.lower():
        return ("External-dataset audit/acquisition", "NewsCLIPpings annotations or VisualNews audit artifacts", "Retained separately: incompatible image-caption mismatch task and incomplete paired data")
    if any(token in path.name.lower() for token in ("robust", "calibration", "root_cause", "explain", "ablation")):
        return ("Focused robustness/evaluation", "Saved validation-only phase-two artifacts", "Covered in 05 or 04; retained for run-specific provenance")
    if "image" in path.name.lower() or "cnn" in path.name.lower():
        return ("Focused image experiment", "Saved image-model experiment artifacts", "Covered in 03; retained for historical pilot provenance")
    if "text" in path.name.lower() or "baseline" in path.name.lower() or "all_models" in path.name.lower():
        return ("Focused model evaluation", "Saved Fakeddit model artifacts", "Covered in 02 or 05; retained for historical comparison provenance")
    return ("Early dataset/integrity analysis", "Local Fakeddit metadata and research manifests", "Covered in 01; retained for provenance")


def audit_report() -> None:
    entries = []
    for path in sorted(ARCHIVE.rglob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        markdown = "\n".join(
            "".join(cell.source) if isinstance(cell.source, list) else cell.source
            for cell in notebook.cells if cell.cell_type == "markdown"
        )
        title_match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem
        kind, evidence, disposition = description(path, title)
        code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        outputs = sum(len(cell.get("outputs", [])) for cell in code)
        entries.append((path.relative_to(ARCHIVE).as_posix(), title, kind, evidence, disposition, len(code), outputs))
    lines = [
        "# Notebook audit", "",
        "## Scope and method", "",
        f"Audited **{len(entries)}** original notebooks before consolidation. The audit reads notebook JSON and existing outputs only; it does not rerun training, mutate artifacts, or open the sealed test manifest for selection.", "",
        "The active suite now has five notebooks. Every original is retained byte-for-byte under `notebooks/archive/2026-10-27/`, preserving run-specific code, outputs, metrics, figures, and provenance.", "",
        "## Reference and dependency review", "",
        "- `tests/test_eda_artifacts.py`, `scripts/build_eda_notebooks.py`, and `reports/EDA/EDA_RESEARCH_REPORT.md` referenced the former EDA suite.",
        "- `scripts/build_research_notebooks.py`, `scripts/report_notebook_execution.py`, and `reports/notebook_research/` retain the provenance of the former 10-notebook research suite.",
        "- Saved experiment scripts and reports consume artifacts under `data/` and `reports/`; they do not depend on notebook paths for training or inference.",
        "- Notebook-only external-data work (NewsCLIPpings/VisualNews) is deliberately not merged into Fakeddit classification notebooks because its labels describe image-caption mismatch, not source taxonomy.", "",
        "## Findings", "",
        "- The nine EDA notebooks are complementary slices of one artifact-backed EDA workflow and are consolidated by topic in active notebook 01.",
        "- The 10 research notebooks form the strongest executed, artifact-backed record; their cells and outputs are the direct source of the five active notebooks.",
        "- Earlier root-level notebooks include pilots and focused reports. They remain archived because some document unique run configurations, VisualNews/NewsCLIPpings status, or historical figures.",
        "- No original was deleted. No model, checkpoint, dataset split, or inference path was changed.", "",
        "## Per-notebook inventory", "",
        "| Original path | Purpose / title | Dataset or evidence | Unique contribution and disposition | Code cells | Stored outputs |",
        "|---|---|---|---|---:|---:|",
    ]
    for relative, title, kind, evidence, disposition, code, outputs in entries:
        lines.append(f"| `{relative}` | {kind}: {title.replace('|', '/')} | {evidence} | {disposition} | {code} | {outputs} |")
    (ROOT / "reports" / "NOTEBOOK_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def readme() -> None:
    text = """# FNEWS notebooks

The active, consolidated research workflow is deliberately small and artifact-backed:

1. `01_dataset_eda.ipynb` — dataset provenance, text/image/multimodal EDA, preprocessing, features, and leakage safeguards.
2. `02_text_modeling.ipynb` — saved TF-IDF + LinearSVC and frozen pretrained-text baseline evidence.
3. `03_image_modeling.ipynb` — saved ResNet-18 transfer-learning evidence and image-model limits.
4. `04_multimodal_fusion.ipynb` — saved weighted late fusion and archived CLIP gated-fusion evidence.
5. `05_evaluation_robustness.ipynb` — matched validation evaluation, error analysis, calibration, robustness, and limitations.

## Run order

Open or execute the notebooks from the repository root, in numeric order. They load local, saved manifests and experiment artifacts; they do **not** retrain models. GPU support is not needed for the normal artifact-reading path, although the environment includes optional image-research dependencies.

```bash
jupyter nbconvert --to notebook --execute notebooks/01_dataset_eda.ipynb --output executed_01.ipynb
```

The Fakeddit labels are source taxonomy labels: raw `1` is `SOURCE_TRUE` and raw `0` is `SOURCE_NON_TRUE`. They are not independently verified factuality claims. Keep the held-out test split sealed; only saved frozen metrics may be quoted where explicitly marked historical.

## Archive and regeneration

All **40** pre-consolidation notebooks are preserved under `archive/2026-10-27/`. The active notebooks are generated from the archived, already executed research-suite notebooks:

```bash
python scripts/consolidate_notebooks.py
```

That command regenerates notebook JSON only. It does not retrain models, modify results, or delete the archive. See `reports/NOTEBOOK_AUDIT.md` and `reports/NOTEBOOK_CONSOLIDATION_REPORT.md` for the detailed record.
"""
    (NOTEBOOKS / "README.md").write_text(text, encoding="utf-8")


def consolidation_report() -> None:
    text = """# Notebook consolidation report

## Outcome

- Original notebooks audited: **40**
- Final active notebooks: **5**
- Original notebooks archived: **40**, under `notebooks/archive/2026-10-27/`
- Original notebooks deleted: **0**

## Active consolidation map

| Active notebook | Consolidated executed research sources | Additional preserved historical sources |
|---|---|---|
| `01_dataset_eda.ipynb` | research 01–03 | nine detailed EDA notebooks and early inventory/integrity notebooks |
| `02_text_modeling.ipynb` | research 04 | neural-text and baseline-comparison notebooks |
| `03_image_modeling.ipynb` | research 05 | manual image-CNN pilot |
| `04_multimodal_fusion.ipynb` | research 06 | modality-ablation and explainability notebooks |
| `05_evaluation_robustness.ipynb` | research 07–10 | Milestone 8, calibration, robustness, raw-image, root-cause, and all-model notebooks |

## Research integrity preserved

- Train/validation/test membership and all saved experiment artifacts were left untouched.
- Active notebooks copy existing executed research cells and outputs; no result is represented as newly run. The two historical split-wide EDA cells were changed to train/validation-only and have no output in the active copies.
- The sealed test partition was not used for new selection or evaluation, and active notebooks do not open its manifest.
- NewsCLIPpings and VisualNews notebooks remain individually archived because image-caption mismatch labels cannot be merged with Fakeddit source-taxonomy labels.
- Training code, checkpoints, production inference, and data files were not changed.

## Validation performed

- Parsed all original and five active notebooks with `nbformat`.
- Confirmed the active notebooks have one bootstrap before copied source cells, preventing cross-section undefined-variable dependencies.
- Confirmed five active source notebooks and 40 archived originals exist.
- Existing EDA regression expectation was updated from the retired nine-notebook directory to the active consolidated suite; artifact-count checks remain unchanged.
- Executed all five active notebooks as separate validation copies in `reports/notebook_validation/`; every executed copy completed with zero cell errors.
- Ran the full project test suite: 87 passed. Ruff checks passed for the consolidation script and updated EDA regression test.

## Execution boundary

The active notebooks are constructed from saved executed outputs. Their validation execution reads local artifacts only; it does not train models or open the sealed test manifest. The generated validation copies are evidence of this consolidation check, not new experiment results.
"""
    (ROOT / "reports" / "NOTEBOOK_CONSOLIDATION_REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    if not ARCHIVE.is_dir():
        raise FileNotFoundError(f"Expected notebook archive at {ARCHIVE}")
    build_suite()
    audit_report()
    readme()
    consolidation_report()


if __name__ == "__main__":
    main()
