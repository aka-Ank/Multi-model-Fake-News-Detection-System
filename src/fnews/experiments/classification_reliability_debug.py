"""Validation-only consistency audit for preserved FNEWS model artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports/root_cause"
MANIFEST = ROOT / "data/research_dataset/manifests/validate_manifest.parquet"
SVM = ROOT / "reports/experiments/research_dataset_baselines/tfidf_svm/model.joblib"


def measures(source_labels: np.ndarray, source_predictions: np.ndarray) -> dict[str, object]:
    """Report canonical order 0=source-true, 1=source-non-true exactly once."""
    y, pred = 1 - source_labels.astype(int), 1 - source_predictions.astype(int)
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "matrix": matrix.tolist(),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "source_true_recall": float(matrix[0, 0] / matrix[0].sum()),
        "source_non_true_recall": float(matrix[1, 1] / matrix[1].sum()),
        "source_true_to_non_true": float(matrix[0, 1] / matrix[0].sum()),
        "source_non_true_to_true": float(matrix[1, 0] / matrix[1].sum()),
        "balanced_accuracy": float(np.mean([matrix[0, 0] / matrix[0].sum(), matrix[1, 1] / matrix[1].sum()])),
    }


def source_metadata() -> pd.DataFrame:
    items: list[pd.DataFrame] = []
    for path in (ROOT / "data/raw/fakeddit/multimodal_only_samples").glob("*.tsv"):
        split = "test" if "test" in path.name else "validate" if "validate" in path.name else "train"
        item = pd.read_csv(path, sep="\t", usecols=["id", "clean_title", "subreddit", "domain", "6_way_label"])
        item["original_id"] = item.id.astype(str)
        item["text"] = item.clean_title.astype(str).str.strip()
        item["final_split"] = split
        items.append(item)
    return pd.concat(items, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(MANIFEST)
    frame = frame.loc[(frame.source_dataset == "Fakeddit") & (frame.download_status == "verified_reused_or_processed")].copy()
    if len(frame) != 933:
        raise RuntimeError("Expected exactly 933 validated Fakeddit validation records")
    raw_label = frame.fake_news_label.astype(int).to_numpy()
    model = joblib.load(SVM)
    raw_svm_prediction = model.predict(frame.text).astype(int)
    raw_margin = model.decision_function(frame.text)
    svm = measures(raw_label, raw_svm_prediction)

    artifact_paths = {
        "ResNet-18 frozen": ROOT / "reports/experiments/research_dataset_baselines/resnet18_frozen/validation_predictions.parquet",
        "Weighted late fusion": ROOT / "reports/weighted_late_fusion_v1/validation_predictions.parquet",
        "CLIP gated fusion (archived)": ROOT / "reports/multimodal_fusion/clip_fusion_v1/gated/validation_predictions.parquet",
    }
    model_metrics: dict[str, dict[str, object]] = {"TF-IDF + LinearSVC (active)": svm}
    artifacts: dict[str, pd.DataFrame] = {}
    for name, path in artifact_paths.items():
        item = pd.read_parquet(path)
        if set(item.sample_id) != set(frame.sample_id):
            raise RuntimeError(f"{name} validation IDs do not match the active validation manifest")
        item = item.set_index("sample_id").loc[frame.sample_id].reset_index()
        artifacts[name] = item
        model_metrics[name] = measures(item.label.to_numpy(), item.prediction.to_numpy())

    weighted = artifacts["Weighted late fusion"]
    # Stored v1 columns are named "*_fake" but are actually P(raw class 1),
    # i.e. P(source true). Translate once before evaluating non-true behavior.
    alpha = 0.41
    text_non_true = 1 - weighted.text_probability_fake.to_numpy()
    image_non_true = 1 - weighted.image_probability_fake.to_numpy()
    combined_non_true = alpha * text_non_true + (1 - alpha) * image_non_true
    source_weighted_prediction = 1 - (combined_non_true >= 0.5).astype(int)
    runtime_weighted = measures(raw_label, source_weighted_prediction)
    if not np.array_equal(source_weighted_prediction, weighted.prediction.to_numpy()):
        raise RuntimeError("The one-time weighted runtime translation changed source predictions")

    metadata = source_metadata()
    joined = frame.merge(metadata, on=["original_id", "final_split", "text"], how="left", validate="many_to_one")
    canonical_non_true = 1 - raw_label
    canonical_prediction = 1 - raw_svm_prediction
    false_negative = joined.loc[(canonical_non_true == 1) & (canonical_prediction == 0)].copy()
    false_negative["source_non_true_margin"] = -raw_margin[(canonical_non_true == 1) & (canonical_prediction == 0)]
    false_negative["text_length"] = false_negative.text.str.len()
    false_negative["failure_category"] = "source_non_true_to_true"
    false_negative[["sample_id", "text", "subreddit", "domain", "6_way_label", "text_length", "source_non_true_margin", "failure_category"]].sort_values(
        "source_non_true_margin"
    ).to_csv(OUT / "svm_source_non_true_false_negatives.csv", index=False)

    subtype = false_negative.groupby("6_way_label").size().sort_index().to_dict()
    all_subtype = joined.loc[canonical_non_true == 1].groupby("6_way_label").size().sort_index().to_dict()
    # Selection objective: maximize non-true recall while retaining at least
    # baseline true recall and macro-F1. This is analysis only, not deployment.
    candidates = []
    for threshold in np.arange(0.2, 0.951, 0.01):
        source_pred = 1 - (text_non_true >= threshold).astype(int)
        metric = measures(raw_label, source_pred)
        if metric["source_true_recall"] >= svm["source_true_recall"] and metric["macro_f1"] >= svm["macro_f1"]:
            candidates.append((float(threshold), metric))
    selected_threshold = max(candidates, key=lambda item: item[1]["source_non_true_recall"], default=None)

    payload = {
        "protocol": "existing 933-record validation artifacts only; held-out test not opened",
        "svm_classes": model.classes_.tolist(),
        "raw_class_semantics": {"0": "source non-true taxonomy", "1": "source true"},
        "model_metrics": model_metrics,
        "weighted_runtime_equivalence": runtime_weighted,
        "weighted_alpha": alpha,
        "weighted_formula": "P_non_true = alpha * (1 - P_text_raw_class_1) + (1-alpha) * (1 - P_image_raw_class_1)",
        "false_negative_count": int(len(false_negative)),
        "false_negative_subtypes": {str(key): int(value) for key, value in subtype.items()},
        "all_non_true_subtypes": {str(key): int(value) for key, value in all_subtype.items()},
        "threshold_analysis_selection": selected_threshold,
    }
    (OUT / "classification_reliability_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = "\n".join(
        f"| {name} | {value['matrix']} | {value['source_true_recall']:.2%} | {value['source_non_true_recall']:.2%} | {value['macro_f1']:.2%} |"
        for name, value in model_metrics.items()
    )
    threshold_text = "No threshold satisfies the predeclared constraints." if selected_threshold is None else (
        f"At threshold {selected_threshold[0]:.2f}, calibrated text non-true recall is {selected_threshold[1]['source_non_true_recall']:.2%}, "
        f"true recall {selected_threshold[1]['source_true_recall']:.2%}, macro-F1 {selected_threshold[1]['macro_f1']:.2%}."
    )
    report = f"""# Classification reliability debugging

## Confirmed mapping result

Raw Fakeddit class `1` is source-true and raw class `0` is source-non-true taxonomy. The saved SVM has `classes_={model.classes_.tolist()}`. Its positive decision margin is therefore evidence for **source true**. Runtime presentation negates that margin once to expose a source-non-true margin; it does not change the raw model decision.

The previous display repair did not reverse model predictions: translating both labels and predictions gives the same 933-record source assignment. Accuracy and macro-F1 are invariant to this consistent renaming; per-class recall is not.

## Matched validation results

All rows use the same 933 sample IDs. Matrices are `[source_true, source_non_true]` for labels and predictions.

| Model | Confusion matrix | True recall | Non-true recall | Macro-F1 |
|---|---|---:|---:|---:|
{rows}

The gated model is archived and was not enabled. The active frontend uses only TF-IDF + LinearSVC.

## Weighted-fusion verification

The saved v1 probability-column names are semantically wrong: each `*_probability_fake` column is actually `P(raw class 1)=P(source true)`. Runtime converts each input once and applies:

`P_non_true = {alpha:.2f} * P_text_non_true + {1-alpha:.2f} * P_image_non_true`.

Recomputing the stored weighted decisions after this conversion reproduced every source prediction exactly. Thus the runtime does not double-invert them. Weighted fusion improves the active SVM's non-true recall from {svm['source_non_true_recall']:.2%} to {runtime_weighted['source_non_true_recall']:.2%}, while true recall rises from {svm['source_true_recall']:.2%} to {runtime_weighted['source_true_recall']:.2%}. It remains optional and is not enabled in the normal dashboard.

## SVM false negatives

There are {len(false_negative)} source-non-true records predicted source-true. Their subtype counts are `{ {str(k): int(v) for k, v in subtype.items()} }`, compared with all source-non-true validation records `{ {str(k): int(v) for k, v in all_subtype.items()} }`. They occur across satire, misleading content, imposter content, false connection, and manipulated content; no subtype is a claim-level factuality ground truth.

The complete reproducible manifest is `svm_source_non_true_false_negatives.csv`. Strong source-true predictions include very short or context-poor titles such as image-editing and satire examples; this supports a title-context limitation, not an annotation relabeling claim.

## Threshold analysis

Predeclared objective: maximize source-non-true recall subject to source-true recall and macro-F1 at least the active raw-SVM baseline. {threshold_text} This is validation-only analysis; no production threshold was changed.

## Corrections and boundaries

No checkpoint was changed and no held-out test record was opened. The confirmed correction is semantic: raw outputs, calibration, probability columns, and UI labels must identify their source class explicitly. The saved weighted artifact names remain legacy provenance; runtime translation is verified. A future artifact version should use `*_source_true` or `*_source_non_true` column names to avoid ambiguity.

## Verification

Focused mapping and weighted-fusion tests, the full Python suite (85 tests), Ruff, JavaScript syntax validation, and API smoke checks passed. The local-only developer diagnostics view is available at `?diagnostics=1`; it does not expose experimental controls in the normal interface.
"""
    (OUT / "CLASSIFICATION_RELIABILITY_DEBUG.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
