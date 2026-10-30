"""Regression checks for the executed, artifact-backed EDA suite."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EDA = ROOT / "reports" / "EDA"


def test_eda_inventory_and_integrity_are_artifact_backed() -> None:
    summary = json.loads((EDA / "eda_summary.json").read_text())
    inventory = pd.read_csv(EDA / "dataset_inventory.csv").set_index("split")
    assert summary["verified_pairs"] == 4233
    assert inventory.loc["train", "verified_pairs"] == 2353
    assert inventory.loc["validate", "verified_pairs"] == 933
    assert inventory.loc["test", "verified_pairs"] == 947
    assert summary["readable_images"] == 4233
    assert summary["corrupt_images"] == 0


def test_eda_notebooks_and_figures_exist() -> None:
    notebooks = sorted((ROOT / "notebooks").glob("0[1-5]_*.ipynb"))
    figures = sorted((EDA / "figures").glob("*.png"))
    assert len(notebooks) == 5
    assert len(figures) >= 20
    assert (EDA / "EDA_RESEARCH_REPORT.md").is_file()
