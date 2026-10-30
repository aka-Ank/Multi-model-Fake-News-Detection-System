# FNEWS

Research-first multimodal fake news detection project using the Fakeddit dataset.

This repository is currently in the dataset preparation and analysis phase. It is intentionally not a production verifier yet and does not train large deep learning models.

## Current Goals

- Set up a modern, reproducible Python project.
- Prepare Fakeddit text, metadata, and image references.
- Run detailed exploratory data analysis.
- Detect dataset quality issues and leakage risks.
- Keep the codebase ready for later text-only, image-only, and multimodal experiments.

## Repository Layout

```text
.
├── configs/                  # Dataset and experiment configuration
├── data/
│   ├── raw/                  # Original TSVs, comments, downloaded archives
│   ├── interim/              # Prepared split manifests and image inventories
│   ├── processed/            # Modeling-ready derived files
│   └── external/             # External evidence/source metadata later
├── docs/                     # Research notes and dataset documentation
├── Fakeddit/                 # Original upstream helper files kept for reference
├── notebooks/                # Exploratory notebooks
├── private_image_set/        # Existing local image archive
├── reports/                  # Generated EDA and leakage reports
├── scripts/                  # Thin command wrappers
├── src/fnews/                # Reusable project code
└── tests/                    # Unit tests
```

## Environment

Use a recent stable Python release. Python 3.11 or 3.12 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The original `Fakeddit/requirements.txt` is preserved for provenance only. It pins very old versions and should not be used for this project environment.

## Data Placement

Place original Fakeddit TSV files under `data/raw/fakeddit/`, preserving their original split names where possible. The code auto-discovers TSV files below that directory.

Examples:

```text
data/raw/fakeddit/all_train.tsv
data/raw/fakeddit/all_validate.tsv
data/raw/fakeddit/all_test_public.tsv
data/raw/fakeddit/multimodal_only_samples/train.tsv
```

Images are expected in `data/raw/fakeddit/images/` by default and should be named by Fakeddit submission ID, for example `awxhir.jpg`. The existing `private_image_set/` folder is preserved, but its hash-style filenames do not match the downloaded public TSV IDs, so it is not treated as paired training data by default.

## Commands

Prepare split manifests and image inventories:

```bash
fnews-prepare-fakeddit --config configs/fakeddit.yaml
```

Generate EDA summaries:

```bash
fnews-eda-fakeddit --config configs/fakeddit.yaml
```

Check duplicate IDs, duplicate text, split overlap, URL overlap, and image overlap:

```bash
fnews-check-fakeddit-leakage --config configs/fakeddit.yaml
```

Download paired images from a TSV file using the `image_url` field:

```bash
fnews-download-fakeddit-images \
  --tsv "data/raw/fakeddit/multimodal_only_samples/multimodal_train.tsv" \
  --output-dir data/raw/fakeddit/images \
  --limit 1000
```

Omit `--limit` only when you intentionally want to download a full split.

The same commands can be run through scripts:

```bash
python scripts/prepare_fakeddit.py --config configs/fakeddit.yaml
python scripts/run_fakeddit_eda.py --config configs/fakeddit.yaml
python scripts/check_fakeddit_leakage.py --config configs/fakeddit.yaml
python scripts/download_fakeddit_images.py --tsv data/raw/fakeddit/multimodal_only_samples/multimodal_train.tsv --limit 1000
```

## Research Notes

For experiments comparable to the original Fakeddit paper, use multimodal samples containing both text and images. Use `clean_title` as the cleaned textual input. See `docs/fakeddit_data_plan.md` for the current preparation plan and leakage checklist.

## Local research website

The repository includes a local-only browser application for manual and article-based research. It uses the saved Fakeddit text and image classifiers when their checkpoint artifacts are present. It does not train models or read the VisualNews archive during startup.

```bash
uv sync --extra dev --extra image-research
cp .env.example .env
set -a && source .env && set +a
uv run fnews-web
```

Open <http://127.0.0.1:8765>. The app stores its SQLite database and uploaded/selected images in `data/research_app/`; these are local application files ignored by Git. Set `FNEWS_APP_DATA` to choose another local data directory. The service refuses public network binding.

The Analyze page supports text-only classification, image-only classification, independent text and image classifiers for a pair, and article extraction/review. Article images are fetched only after the user selects a candidate for preview. See [the implementation plan](docs/IMPLEMENTATION_PLAN.md), [architecture](docs/ARCHITECTURE.md), and [progress](docs/PROGRESS.md) for verified model boundaries, security behavior, and remaining work.
# Multi-model-Fake-News-Detection-System
