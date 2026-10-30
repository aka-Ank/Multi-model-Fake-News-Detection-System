# SVC from Scratch

This directory is an independent educational research package. It does not import, modify, or replace FNEWS's existing classifier. Its core algorithms use only the Python standard library.

## Run

From this folder, run:

```bash
python3 run_tests.py
python3 experiments.py
```

The first command runs the automated mathematical checks. The second writes measured synthetic results to `reports/synthetic_results.json`. It does not access or alter FNEWS data.

## Scope

- `src/svm_primal.py`: linear soft-margin primal SVM with mini-batch subgradients.
- `src/svm_dual.py`: binary kernel SVM using SMO.
- `src/kernels.py`: linear, polynomial, and RBF kernels.
- `src/text_vectorizer.py`: train-only TF-IDF with optional sparse dictionaries.
- `src/metrics.py`: manually implemented classification metrics.

`SVC` maps two original labels to `-1/+1` only while optimizing and maps predictions back afterward. Decision-function outputs are signed, uncalibrated margins or scores, never probabilities.

## Optional FNEWS text experiment

The main experiments are synthetic by design. FNEWS's verified paired manifests are Parquet and therefore cannot be read by a standard-library-only core. `fnews_fakeddit_tsv_experiment` reads a supplied official Fakeddit TSV with the standard library, first verifying `1 = SOURCE_TRUE` through six-way class 0 and `0 = SOURCE_NON_TRUE` through six-way classes 1–5. It then reservoir-samples a fixed-size, read-only subset and fits TF-IDF only on its training partition. `optional_tsv_experiment` accepts a different user-prepared TSV with explicit `text` and `label` columns.

The dense SVC is deliberately bounded. A dense float list can require roughly `8 * rows * features` bytes before Python-object overhead, while the dual kernel matrix requires O(n²) values. Use small experiments and keep `max_features` low.
