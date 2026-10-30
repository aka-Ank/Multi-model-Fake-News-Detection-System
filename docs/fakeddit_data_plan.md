# Fakeddit Data Plan

## Scope

This phase prepares the Fakeddit dataset for research experiments. It does not build the final production misinformation system and does not train large deep learning models.

## Canonical Inputs

- Text input: `clean_title`
- Image input: local image file matched by submission `id`, or `image_url` when downloading is needed
- Labels: available Fakeddit label columns such as `2_way_label`, `3_way_label`, and `6_way_label`
- Paper-comparable subset: samples with both text and image, especially files under `multimodal_only_samples`

## Preparation Checklist

1. Preserve raw files unchanged under `data/raw/fakeddit/`.
2. Keep the original upstream helper files under `Fakeddit/` for provenance.
3. Normalize split discovery into manifests under `data/interim/fakeddit/`.
4. Record image presence using local image inventory before downloading anything.
5. Prefer `clean_title` for all first-pass text analysis.
6. Keep comments and external evidence out of baseline text-only experiments unless explicitly studying those signals.

## EDA Checklist

- Row counts by split and file.
- Label distributions for 2-way, 3-way, and 6-way labels.
- Missingness across text, images, URLs, labels, and metadata.
- Text length distributions from `clean_title`.
- Multimodal coverage by split.
- Domain/source patterns from URL columns when present.
- Subreddit or author concentration if metadata columns exist.

## Leakage and Quality Checklist

- Duplicate IDs within and across splits.
- Duplicate `clean_title` values across splits.
- Identical image IDs or local image filenames across splits.
- URL overlap across splits.
- Near-identical text can be added later with MinHash or embedding-based checks.
- Verify that multimodal-only experiments do not accidentally include text-only or image-only rows.

## Later Experiment Tracks

- Text-only baselines using `clean_title`.
- Image-only baselines using local images.
- Multimodal baselines using paired text and images.
- Source and URL-based metadata features.
- External evidence retrieval and verification.
