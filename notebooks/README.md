# FNEWS notebooks

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

That command regenerates notebook JSON only. It does not retrain models, modify results, or delete the archive. Generated audit reports are kept locally and are not part of this source repository.
