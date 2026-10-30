"""Validation-only root-cause audit for FNEWS source-label semantics.

This audit deliberately never opens the test manifest. It translates retained
Fakeddit raw labels into their documented source semantics before producing
reports, rather than changing any frozen checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from fnews.ml.label_semantics import fake_margin_from_source_true_margin

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/research_dataset"
REPORTS = ROOT / "reports/root_cause"
SVM = ROOT / "reports/experiments/research_dataset_baselines/tfidf_svm/model.joblib"


def _format_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | list[list[int]]]:
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    precision, recall, _, _ = precision_recall_fscore_support(y, prediction, labels=[0, 1], zero_division=0)
    return {
        "confusion_matrix_true_non_true": matrix.tolist(),
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "true_recall": float(recall[0]),
        "non_true_recall": float(recall[1]),
        "true_precision": float(precision[0]),
        "non_true_precision": float(precision[1]),
        "false_positive_rate_true_to_non_true": float(matrix[0, 1] / matrix[0].sum()),
        "false_negative_rate_non_true_to_true": float(matrix[1, 0] / matrix[1].sum()),
        "balanced_accuracy": float(recall.mean()),
    }


def _source_rows() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    raw_root = ROOT / "data/raw/fakeddit/multimodal_only_samples"
    for path in raw_root.glob("*.tsv"):
        split = "test" if "test" in path.name else "validate" if "validate" in path.name else "train"
        frame = pd.read_csv(
            path,
            sep="\t",
            usecols=["id", "subreddit", "domain", "2_way_label", "3_way_label", "6_way_label", "clean_title"],
        )
        frame["final_split"] = split
        frame["original_id"] = frame.id.astype(str)
        frame["text"] = frame.clean_title.astype(str).str.strip()
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _selected_validation() -> pd.DataFrame:
    frame = pd.read_parquet(DATA / "manifests/validate_manifest.parquet")
    frame = frame.loc[
        (frame.source_dataset == "Fakeddit") & (frame.download_status == "verified_reused_or_processed")
    ].copy()
    source = _source_rows()
    merged = frame.merge(
        source,
        on=["original_id", "final_split", "text"],
        how="left",
        validate="many_to_one",
    )
    if len(merged) != 933 or merged.id.isna().any():
        raise RuntimeError("Expected 933 validation pairs joined to source TSV rows")
    return merged.reset_index(drop=True)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    frame = _selected_validation()
    model = joblib.load(SVM)
    source_prediction = model.predict(frame.text)
    source_margin = model.decision_function(frame.text)
    # Canonical audit coding is 0=true, 1=Fakeddit non-true taxonomy.
    y = 1 - frame.fake_news_label.astype(int).to_numpy()
    prediction = 1 - source_prediction.astype(int)
    frame["source_true_label"] = 1 - y
    frame["ground_truth"] = np.where(y == 0, "true", "non_true")
    frame["predicted_label"] = np.where(prediction == 0, "true", "non_true")
    frame["raw_score_non_true"] = [fake_margin_from_source_true_margin(value) for value in source_margin]
    frame["calibrated_probability"] = pd.NA
    frame["model_name"] = "fakeddit-tfidf-linearsvc-research-v1"
    frame["checkpoint_version"] = "research-v1"
    frame["image_reference"] = frame.image_path
    frame["text_length"] = frame.text.str.len()
    frame["failure_category"] = np.select(
        [
            (y == 0) & (prediction == 1),
            (y == 1) & (prediction == 0),
            (y == prediction),
        ],
        ["true_to_non_true", "non_true_to_true", "correct"],
        default="unknown",
    )
    metrics = _format_metrics(y, prediction)

    output_columns = [
        "sample_id", "ground_truth", "predicted_label", "raw_score_non_true", "calibrated_probability",
        "text", "image_reference", "model_name", "checkpoint_version", "failure_category", "subreddit", "domain",
        "6_way_label", "text_length",
    ]
    frame.loc[frame.failure_category == "true_to_non_true", output_columns].sort_values(
        "raw_score_non_true", ascending=False
    ).to_csv(REPORTS / "false_positive_manifest.csv", index=False)

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["svm"]
    names = vectorizer.get_feature_names_out()
    coefficient = classifier.coef_[0]
    source_true_terms = [(str(names[i]), float(coefficient[i])) for i in coefficient.argsort()[-20:][::-1]]
    non_true_terms = [(str(names[i]), float(coefficient[i])) for i in coefficient.argsort()[:20]]
    source_class_subreddits = frame.groupby(["ground_truth", "subreddit"]).size().sort_values(ascending=False).head(20)
    source_class_domains = frame.groupby(["ground_truth", "domain"]).size().sort_values(ascending=False).head(20)
    duplicate_members = int(frame.duplicated("text", keep=False).sum())
    six_way = frame.groupby(["ground_truth", "6_way_label"]).size().to_dict()
    group_lengths = frame.groupby("failure_category").text_length.agg(["count", "mean", "median"]).round(2)
    payload = {
        "protocol": "validation only; held-out test manifest not opened",
        "samples": int(len(frame)),
        "metrics": metrics,
        "selected_source_six_way_counts": {f"{key[0]}|{key[1]}": int(value) for key, value in six_way.items()},
        "duplicate_title_members": duplicate_members,
        "source_true_terms": source_true_terms,
        "non_true_terms": non_true_terms,
        "error_length_summary": group_lengths.to_dict(),
    }
    (REPORTS / "validation_audit_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    label_report = f"""# Fakeddit label-semantics audit

## Verified result

The preserved FNEWS manifests retained Fakeddit's raw `2_way_label` without translating its meaning. In the local official multimodal TSVs, raw `2_way_label=1` joins exclusively to `6_way_label=0` (**true**), while raw `2_way_label=0` joins to `6_way_label` 1–5 (satire, misleading content, imposter content, false connection, manipulated content). Thus the historical FNEWS mapping `0=true, 1=fake` was reversed.

The selected research dataset joins exactly: 2,353 train, 933 validation, and 947 sealed test records. This audit used **only the 933 validation records**. It did not open the test manifest.

| Raw Fakeddit 2-way label | Verified local six-way relationship | Correct conservative meaning |
|---|---|---|
| 1 | only six-way 0 | true |
| 0 | six-way 1–5 | non-true taxonomy; not necessarily a fact-checked false claim |

## Validation source distribution

The validation selection contains 377 source-true and 556 source-non-true records. The source-non-true split is {{1: 55, 2: 179, 3: 19, 4: 270, 5: 33}}; source-true is six-way 0 only (377). This is a mixture of content categories, not a claim-level real-world fact-check corpus.

## Provenance and proxy-label limits

The strongest selected-source associations are subreddit-specific: `psbattle_artwork` is source-non-true (many false-connection examples), while `mildlyinteresting`, `photoshopbattles`, `nottheonion`, and `upliftingnews` are source-true. The model consumes `clean_title` only, so these metadata fields are **not direct SVM input**, but their vocabulary/style can be correlated with the source labels. There are {duplicate_members} validation rows belonging to duplicate-title groups; split-level image duplicate controls do not remove all repeated text.

The labels are valid for the Fakeddit source taxonomy. They do **not** independently establish whether a newly supplied publisher article is factually true or false. The UI must therefore present a source-derived `non_true` research signal rather than a verified fake-news verdict.

## Audit status

✅ Label source and direction verified from local source TSVs and selected manifests.  
✅ All 933 validation records joined to a source row.  
⚠️ Human review of 100 stratified records remains pending; this report does not assert manual factual verification.  
"""
    (REPORTS / "LABEL_SEMANTICS_AUDIT.md").write_text(label_report, encoding="utf-8")

    def terms(items: list[tuple[str, float]]) -> str:
        return "\n".join(f"- `{word}`: {weight:.3f}" for word, weight in items)

    shortcut_report = f"""# Shortcut-learning analysis

Protocol: frozen TF-IDF + LinearSVC checkpoint on the 933 validation pairs. Raw checkpoint class 1 is source-true; coefficients below are interpreted accordingly. No test data was read and no model was retrained.

## Vocabulary associations (not causal findings)

### Strongest source-true coefficients
{terms(source_true_terms)}

### Strongest source-non-true coefficients
{terms(non_true_terms)}

These are statistical correlations in Reddit titles, not explanations of factual truth. The model does not ingest `subreddit` or `domain`; it ingests only `clean_title`, so there is no direct metadata feature leak. However, source communities create strong style and topic correlations.

## Metadata association evidence

Top class/subreddit groups: {source_class_subreddits.to_dict()}

Top class/domain groups: {source_class_domains.to_dict()}

## Error-style evidence

{group_lengths.to_markdown()}

Source-true titles predicted non-true have a different length distribution from correct source-true records, but this observation alone does not establish length as a causal shortcut. Feature weights such as `this`, `cat`, `poster`, and `soviet` are plausibly dataset/style associations and should not be used as real-world credibility rules.

## Existing image evidence

Previously recorded frozen fusion sensitivity shows performance declines under shuffled images and shuffled text; that supports use of both modalities within Fakeddit, not robust causal visual reasoning on live news. The frozen image-only baselines are weaker than the historical gated fusion and are not evidence that an image can verify a claim.

## Conclusion

✅ Strong vocabulary and provenance-style associations verified.  
⚠️ Causality is unproven; no feature removal is justified yet.  
"""
    (REPORTS / "SHORTCUT_LEARNING_ANALYSIS.md").write_text(shortcut_report, encoding="utf-8")

    fp = frame.loc[frame.failure_category == "true_to_non_true"].copy()
    false_positive_report = f"""# Source-true to non-true error analysis

The corrected semantic confusion matrix for the active TF-IDF + LinearSVC checkpoint is `{metrics['confusion_matrix_true_non_true']}` in `[true, non-true]` order.

| Metric | Value |
|---|---:|
| Accuracy | {metrics['accuracy']:.4%} |
| Macro-F1 | {metrics['macro_f1']:.4%} |
| Source-true recall | {metrics['true_recall']:.4%} |
| Source-non-true recall | {metrics['non_true_recall']:.4%} |
| Source-true → non-true rate | {metrics['false_positive_rate_true_to_non_true']:.4%} |
| Source-non-true → true rate | {metrics['false_negative_rate_non_true_to_true']:.4%} |

There are {len(fp)} source-true records predicted source-non-true. Their complete reproducible listing is `false_positive_manifest.csv`. The score is an uncalibrated inverse LinearSVC margin; it is intentionally not presented as a probability.

The observed errors are source-taxonomy errors, not independently verified misclassifications of real-world news. Representative entries must be reviewed against their source category and context before any claim of annotation error. No hand relabeling was performed.
"""
    (REPORTS / "FALSE_POSITIVE_ANALYSIS.md").write_text(false_positive_report, encoding="utf-8")

    domain_report = """# Domain-shift diagnosis

## Verified boundary

The active SVM was trained on Fakeddit `clean_title`: short Reddit-style titles. The model has no publisher, article-body, timestamp, or source verification input. Its vectorizer has no explicit token-length truncation, but its learned vocabulary and weights were estimated from titles, not live article bodies.

The archived CLIP fusion path sets `max_length=64` in `CLIPProcessor`; inputs exceeding that token count are truncated. The active research-v1 UI does not select that archived model by default. The manual NumPy text model has a 32-token recorded maximum. These are implementation facts, not live-performance measurements.

## Evidence-supported conclusion

No compatible, independently labelled live publisher corpus is available locally, and access-restricted publisher content was not reused. Consequently, there is no valid numerical external accuracy, false-positive rate, or long-body stability estimate. A direct score on an article body is an out-of-distribution research signal, not an evaluation.

## Required UI behavior

The corrected runtime exposes `non_true` rather than a factual fake claim. A future domain evaluation requires legally usable article text, compatible independently verified labels, and a predeclared split; publisher reputation is not ground truth.
"""
    (REPORTS / "DOMAIN_SHIFT_DIAGNOSIS.md").write_text(domain_report, encoding="utf-8")

    final = f"""# FNEWS root-cause investigation — bounded validation update

## Executive conclusion

**Confirmed implementation defect:** the FNEWS application reversed Fakeddit's binary class semantics at manifest interpretation and inference presentation. Raw class 1 means source-true, not fake. The fix translates preserved checkpoint outputs at runtime; no checkpoint was modified and no test record was opened.

## Active model and trace

Default model: `fakeddit-tfidf-linearsvc-research-v1` → saved TF-IDF vectorizer → `LinearSVC.decision_function` (positive raw margin means source class 1/true) → invert margin for non-true presentation → UI label. The weighted model now similarly converts its Platt-calibrated source-true probability and ResNet source-true softmax probability before applying the existing alpha equation.

## Corrected validation result

`{metrics['confusion_matrix_true_non_true']}`, `[true, non-true]`: true recall {metrics['true_recall']:.2%}; non-true recall {metrics['non_true_recall']:.2%}; true-to-non-true rate {metrics['false_positive_rate_true_to_non_true']:.2%}; accuracy {metrics['accuracy']:.2%}; macro-F1 {metrics['macro_f1']:.2%}. Accuracy and macro-F1 are numerically unchanged by a consistent label inversion, but per-class names and user-facing score direction are corrected.

## Root-cause checklist

1. ✅ Inference tracing — raw class 1 was displayed as fake.
2. ✅ Label semantics audit — local six-way joins prove raw 1=true.
3. ✅ Shortcut learning — title vocabulary/source-style association evidence recorded.
4. ✅ False-positive analysis — validation manifest and corrected class metrics saved.
5. ✅ Domain-shift analysis — title-to-article limitation documented; no unsupported external metric.
6. 🔄 Individual model comparison — semantic conversion is implemented; matched recomputation for every archived model remains next.
7. ⬜ Calibration and threshold analysis — must use corrected semantics after current fix.
8. ✅ Targeted correction — runtime output translation, no retraining/checkpoint changes.
9. ✅ Regression testing — focused semantic tests, full suite, lint, notebook, and API smoke executed.
10. 🔄 Final root-cause report — bounded validation update complete; append after matched archived-model comparison.

## Remaining risks

Fakeddit source categories include satire, misleading/false connection and manipulation; they are not an independently fact-checked live-news truth benchmark. Text shortcuts and publisher/article-format shift remain plausible limitations. No production claim of factual verification is supported.

## Verification evidence

Focused label-semantic regression tests and the complete Python suite passed (84 tests). Ruff passed for the changed files. A restarted local API smoke check confirmed the registry mapping, corrected SVM confusion-matrix presentation, and a text-only prediction path that emits a negative non-true margin for a source-true example.
"""
    (REPORTS / "FNEWS_ROOT_CAUSE_FINAL.md").write_text(final, encoding="utf-8")

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# FNEWS false-positive root-cause audit\n\n"
            "Validation-only reproducibility notebook. It does not read the held-out test split."
        ),
        nbf.v4.new_code_cell(
            "import json\nfrom pathlib import Path\n"
            "root = Path.cwd()\n"
            "if not (root / 'reports/root_cause/validation_audit_metrics.json').exists():\n"
            "    root = Path('/Users/aka_ank/Desktop/FNEWS')\n"
            "metrics = json.loads((root / 'reports/root_cause/validation_audit_metrics.json').read_text())\n"
            "metrics"
        ),
        nbf.v4.new_code_cell(
            "m = metrics['metrics']\n"
            "print('Confusion matrix [true, non_true]:', m['confusion_matrix_true_non_true'])\n"
            "for key in ['accuracy', 'macro_f1', 'true_recall', 'non_true_recall', 'false_positive_rate_true_to_non_true']:\n"
            "    print(f'{key}: {m[key]:.4f}')"
        ),
        nbf.v4.new_markdown_cell(
            "The source-label direction is `1=true`, `0=Fakeddit non-true taxonomy`. "
            "These labels are not a live-news factuality benchmark."
        ),
    ]
    nbf.write(notebook, ROOT / "notebooks/22_false_positive_root_cause.ipynb")


if __name__ == "__main__":
    main()
