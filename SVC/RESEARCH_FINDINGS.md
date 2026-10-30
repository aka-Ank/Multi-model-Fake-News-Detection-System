# Research Findings

## Objective and method

This study implements binary support-vector classification from first principles using only Python's standard library. The package has separate primal and dual/SMO paths, manual kernels, TF-IDF, metrics, and unit tests. The default experiment uses deterministic synthetic data so results can be independently rerun without silently changing FNEWS data.

## Experimental setup

`experiments.py` measures a linearly separable two-dimensional set with primal and dual linear SVMs, then measures XOR with linear and RBF dual SVCs. It also sweeps `C`, RBF `gamma`, learning rate, and requested epochs on deliberately tiny deterministic sets. It records accuracy, macro-F1, diagnostics, runtime, and memory estimates in a JSON artifact. Results are populated only by an executed run.

## Executed results

The executed synthetic run recorded 100% accuracy and macro-F1 for both linear optimizers on the separable six-point set. On XOR, the linear dual model obtained 50% accuracy and the RBF model with `C=10, gamma=2` obtained 100%. This verifies a representational difference on a known geometry, not a fake-news performance result. The complete measured sensitivity values are in `reports/synthetic_results.json`.

The read-only Fakeddit TSV experiment verified 36,022 raw `0 = SOURCE_NON_TRUE` rows and 23,320 raw `1 = SOURCE_TRUE` rows against the six-way taxonomy. Its deterministic reservoir supplied 600 examples; 475 were used to fit a 300-term train-only TF-IDF vocabulary and 125 to assess the scratch primal SVC. The held-out demonstration partition measured 0.616 accuracy, 0.615 macro-F1, and 0.616 balanced accuracy, with confusion matrix `[[42, 20], [28, 35]]` in `[SOURCE_NON_TRUE, SOURCE_TRUE]` order. It stopped after 164 epochs with 364 training margin violations. This is an educational, bounded source-taxonomy result, not a comparison to the existing FNEWS baseline or a real-world fake-news claim.

## Convergence and stability

The primal method records its exact objective after each epoch and stops only when successive objectives meet its tolerance. SMO records iterations, equality residual, support-vector count, and remaining KKT violations. An iteration limit is an explicit safeguard, not proof of convergence. Floating-point tolerance, alpha clipping, kernel curvature, and near-zero eta are exposed design choices.

## Complexity and limitations

Primal linear training is O(n d) per epoch with O(d) model storage. The dual kernel method constructs O(n²) kernel storage and requires repeated pair optimization, so it is intentionally restricted to small datasets. The custom TF-IDF offers sparse dictionaries, but SVC experiments densify features to retain a simple, auditable numerical implementation. This is unsuitable for full-scale Fakeddit training.

## FNEWS interpretation

The optional text route must preserve FNEWS's verified research-task mapping: raw `1 = SOURCE_TRUE`, raw `0 = SOURCE_NON_TRUE`; conversion to signed labels occurs only inside optimization. These labels describe source taxonomy, not independently verified truth. The experiment must never pool NewsCLIPpings mismatch labels with Fakeddit source labels.

## Future directions

Useful extensions are a sparse linear optimizer, multiclass one-vs-rest with carefully preserved label semantics, held-out validation protocols, and an optional separate reference-comparison notebook after the independent tests pass. None are required for this binary, from-scratch study.
