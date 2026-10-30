# Research roadmap

1. Use `data/research_dataset/manifests/` for Fakeddit image-text fake-news experiments. It contains only validated local pairs.
2. Keep NewsCLIPpings out of multimodal training until the official VisualNews metadata maps caption and image IDs and the referenced source images can be acquired individually.
3. If that source becomes available, run a bounded source-ID-based acquisition into the same dataset root, then rebuild the manifests and leakage report. Do not download `origin.tar` solely to reach this step.
4. Train task-specific heads with `fake_news_label` and `image_caption_mismatch_label` independently; evaluate each task on its preserved official split.
5. Use the executed notebooks for dataset review before any model-selection decision. Do not tune against test data.
6. URL analysis now ranks structured-data, Open Graph, Twitter, and article-body images without bypassing SSRF, robots, content-type, size, or image-decode checks. Maintain the explicit `multimodal` versus `text_only` mode boundary and retain the selected source URL and ranking reason in every reviewed pair.
7. Keep the deployed gated-fusion checkpoint fixed while collecting out-of-distribution URL workflow observations. Any future recalibration, retraining, or mismatch detector must use a new frozen protocol and must not tune against the held-out Fakeddit test partition.
8. Phase 2 baseline audit is complete in `reports/phase2/RESEARCH_BASELINE.md`. Resume with a bounded external-dataset compatibility review; this is currently blocked locally by the lack of verified external image-text fake-news pairs. Do not treat NewsCLIPpings mismatch annotations as fake-news supervision.
9. The ReCOVery compatibility review and 100-record pilot are complete. Its 66 validated pairs retain source-reliability labels and do not support external Fakeddit-style accuracy. Frozen validation-only modality analysis is complete; resume with probability calibration or explainability, never held-out-test tuning.
10. Temperature calibration is assessed on a stratified validation-half protocol and shown beside raw fusion scores without changing labels or checkpoint weights. Resume with a registered raw-image robustness/attribution protocol; do not call current gate diagnostics or shuffles causal explanations.
11. Root-cause audit verified that local Fakeddit `2_way_label=1` is source-true and `0` is the source non-true taxonomy. Runtime output translation is verified without changing checkpoints; use `reports/root_cause/` for corrected class-wise validation interpretation and do not represent the result as external factuality verification.
12. The user-facing dashboard is simplified to the active TF-IDF + LinearSVC policy with Article URL and Text + image tabs. Optional images are contextual only under this policy; experimental model controls are hidden, and the UI displays `SOURCE_TRUE` / `SOURCE_NON_TRUE` rather than factuality verdicts.
13. The final-year research notebook suite (`notebooks/research/01` through `10`) is complete and executed from saved artifacts: 10/10 notebooks, 13 embedded figures, and zero cell errors. The final notebook contains EDA totals, a result diagram, and training-method summary. It does not retrain models or use the held-out test partition for selection.
14. Comprehensive EDA is complete in `notebooks/EDA/01` through `09` and `reports/EDA/EDA_RESEARCH_REPORT.md`: 4,233 verified pairs, 20 data-derived figures, and zero notebook cell errors. It confirms 19 exact-text groups (140 records) spanning splits, no exact image-hash cross-split groups, and zero corrupt processed images. Preserve the frozen split; use a separate grouped/deduplicated protocol for future experiments.

## PHASE STATUS CHECKLIST

| Phase | Status | Evidence / current metrics | Next action |
| --- | --- | --- | --- |
| Validated Fakeddit dataset | ✅ Completed | 4,233 pairs; 2,353/933/947 split; leakage report; 0.158 GiB | Preserve split and cap |
| Frozen gated fusion | ✅ Completed | Validation 87.46% accuracy / 87.07% macro-F1; historical test 86.91% / 86.45% | Never tune on test |
| Secure URL inference | ✅ Completed | Controlled workflow plus verified source-label runtime translation; 87 tests and API smoke passed | Keep explicit `non_true` research-label warning |
| External acquisition | ✅ Completed | Five-source audit; ReCOVery pilot 66 valid / 100 selected | Keep labels source-native |
| External zero-shot evaluation | ⏸️ Deferred by user | Explicitly excluded from current scope; previous limitations preserved | Do not acquire or evaluate externally |
| Modality contribution | ✅ Completed | Shuffled image/text macro-F1: 70.77% / 63.82% | Treat as sensitivity only |
| Probability calibration | ✅ Completed | Assessment ECE 0.0889→0.0380; Brier 0.1017→0.0898 | Monitor transfer |
| Explainability | 🔄 In progress | Gate/norm/perturbation diagnostics executed | Raw image/token attribution |
| Robustness | 🔄 In progress | 16-pair pilot plus 32-pair/19-condition raw-image sweep; blur sigma 3 macro-F1 77.03% in expanded subset | Full partition or registered text protocol |
| Live-publisher URL QA | ⚠️ Blocked | Reuters/AP access denial; Guardian Developer terms reviewed and prohibit ML/AI analysis of API content | Resume only with a source explicitly authorizing ML inference |
| Mismatch detector | ⏸️ Deferred by user | Explicitly excluded from current scope; zero valid pairs preserved | Do not acquire VisualNews or train mismatch model |
| Model improvement | ✅ Completed | Weighted late fusion retained as optional; source-label direction corrected without checkpoint changes | Recompute matched class metrics before any promotion |
| Final reproducibility review | 🔄 In progress | Initial audit and checkpoint verification | Final synthesis after feasible phases |
| Comprehensive EDA | ✅ Completed | 9/9 notebooks executed, 20 figures; 4,233 pairs; 19 exact-text cross-split groups; 0 corrupt images | Preserve baseline split and use a future grouped protocol |

All future FNEWS progress reports must include this checklist format. A phase is **✅ Completed** only after implementation, execution, and verification—not merely code creation. External zero-shot evaluation and mismatch detection are explicitly **⏸️ Deferred by user** and excluded from the active-completion denominator (11 active phases).
