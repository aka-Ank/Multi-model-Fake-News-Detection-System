# FNEWS: Multimodal Fake News Research

FNEWS is a research project built around the public [Fakeddit dataset](https://fakeddit.netlify.app/). It includes data-preparation tools, five active Jupyter notebooks, text and image experiments, saved model checkpoints, and a local browser application for exploring model predictions.

The classifiers learn Fakeddit's source/community label taxonomy. They do not verify whether a claim is factually true, and their dataset results should not be read as real-world fact-checking accuracy.

## What is included

- `notebooks/` — active dataset, text, image, multimodal, and evaluation notebooks, plus archived research notebooks.
- `src/fnews/` — reusable data, modeling, experiment, and local web application code.
- `scripts/` — command-line wrappers and notebook/report helpers.
- `configs/` — dataset and experiment configurations.
- `tests/` — automated checks for the project code.
- `reports/` — selected small checkpoints and metric files required by the website. Dataset copies, generated reports, and other experiment outputs stay local.
- `docs/` — architecture, research roadmap, data plan, and method notes.
- `SVC/` — a separate pure-standard-library SVC-from-scratch research implementation.

## Quick start

Requirements: Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/aka-Ank/Multi-model-Fake-News-Detection-System.git
cd Multi-model-Fake-News-Detection-System
uv sync --extra dev --extra image-research
```

### Run the local research website

```bash
uv run fnews-web
```

Open <http://127.0.0.1:8765>. The application is for local use and binds to loopback by default. It includes text-only and image-only model analysis, article text extraction for user review, and a local history backed by SQLite. The app stores its database and uploaded images under `data/research_app/`; those files are excluded from Git. Optional settings are documented in [`.env.example`](.env.example).

The repository contains the compact checkpoints needed for the listed website models. Some external model families and fusion experiments are unavailable or inactive; the interface reports their availability instead of claiming every experiment is deployable.

### Open the notebooks

```bash
uv run jupyter lab
```

The active notebooks are:

1. `notebooks/01_dataset_eda.ipynb`
2. `notebooks/02_text_modeling.ipynb`
3. `notebooks/03_image_modeling.ipynb`
4. `notebooks/04_multimodal_fusion.ipynb`
5. `notebooks/05_evaluation_robustness.ipynb`

The Fakeddit data is not included. Download it from the [dataset release](https://drive.google.com/drive/folders/1jU7qgDqU1je9Y0PMKJ_f31yXRo5uWGFm?usp=sharing), review its usage terms, and place the TSV files under `data/raw/fakeddit/`. The default paths and preparation workflow are described in [`docs/fakeddit_data_plan.md`](docs/fakeddit_data_plan.md) and [`configs/fakeddit.yaml`](configs/fakeddit.yaml).

### Prepare and inspect data

```bash
uv run fnews-prepare-fakeddit --config configs/fakeddit.yaml
uv run fnews-eda-fakeddit --config configs/fakeddit.yaml
uv run fnews-check-fakeddit-leakage --config configs/fakeddit.yaml
```

These commands read local dataset files and write derived data or reports to ignored local directories. See the configuration and data plan before running them.

## Included model artifacts

The website's inference code loads saved artifacts from `reports/`:

- A manually implemented TF-IDF + LinearSVM text classifier.
- A NumPy text embedding classifier.
- A small NumPy CNN image pilot.
- A ResNet-18 image checkpoint and calibrator used by the optional weighted fusion entry.

The artifacts and their metadata/metrics are included so the application can load the corresponding models after setup. The image CNN is a preliminary pilot, and multimodal fusion models do not all have deployable checkpoints. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for application behavior and [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md) for experiment status and limits.

## Data and Git exclusions

Dataset files, private/local image collections, virtual environments, caches, credentials, Codex/AI assistant state, and most generated reports are excluded by [`.gitignore`](.gitignore). The ignore file explicitly allows only the compact checkpoints and metrics needed for the website. Keep `.env` local; `.env.example` is safe to share.

## Citation

If you use Fakeddit, cite:

> Kai Nakamura, Sharon Levy, and William Yang Wang. 2020. “r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection.” *Proceedings of EMNLP 2020.* <https://arxiv.org/abs/1911.03854>
