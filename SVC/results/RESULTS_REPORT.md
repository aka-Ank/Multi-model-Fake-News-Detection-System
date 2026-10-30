# SVC from Scratch: research results

## 1. Research objective and implementation

This study validates a binary, from-scratch SVC implementation. It includes a linear soft-margin primal subgradient solver, a dual kernel SMO solver, linear/polynomial/RBF kernels, manual TF-IDF, and manual classification metrics. Core SVC files are standard-library-only; matplotlib and scikit-learn are isolated in optional reporting/reference files.

## 2. Mathematical validation

**Observation:** finite differences agree with the implemented primal subgradient. **Evidence:** maximum absolute error was `1.59e-10` away from the hinge boundary. **Interpretation:** the objective, regularization, and active-hinge gradient agree numerically at the tested point. **Limitation:** a finite test does not prove all numerical paths. **Conclusion:** the tested primal derivative is supported.

The linear dual validation ended with dual objective `0.15999999`, equality residual `0`, box violation `0`, maximum KKT residual `1.79e-4`, and three support vectors. This meets the selected `1e-3` tolerance. See [mathematical validation](MATHEMATICAL_VALIDATION.md).

## 3. Synthetic experiments

On held-out separable Gaussian data, both primal and dual linear models reached accuracy and macro-F1 of `1.000`. On noisy held-out XOR, linear SMO reached `0.650` accuracy / `0.649` macro-F1, while polynomial degree-2 and RBF models each reached `1.000` / `1.000`. A linear boundary cannot separate XOR because its alternating labels occupy opposing corners; the nonlinear kernels map it to a separable representation.

On the held-out imbalanced-overlap set, linear primal reached `0.833` accuracy but only `0.455` macro-F1. **Interpretation:** accuracy masked minority-class performance. **Conclusion:** macro-F1 and balanced accuracy are required beside accuracy for class-imbalanced studies.

## 4. Optimization and hyperparameters

Primal trace plots show non-monotone mini-batch behavior, which is expected for subgradient updates. The dual trace records objective, KKT residual, alpha updates, equality residual, and support-vector count per pass. The degree-2 polynomial XOR run stopped at `max_iter` with residual `1.42e-4`; this is reported as incomplete despite perfect held-out predictions.

Validation-only C sweeps on overlap data yielded held-out macro-F1 from `0.448` to `0.519`; no test metric chose C. RBF gamma sweeps on XOR showed low-gamma underfitting (`0.231` macro-F1 at gamma 0.001) and strong performance at gamma 1 and 10. The complete values are in [the experiment table](tables/experiment_summary.csv).

## 5. Primal, dual, and reference comparison

On a separate 50/25 Gaussian-overlap reference split, custom primal, custom dual linear, and sklearn LinearSVC all reached `1.000` macro-F1 with prediction agreement `1.000` to custom primal. Custom and sklearn RBF each reached `0.959` macro-F1 and `0.960` agreement. These are sanity checks, not claims of parameter equivalence: objective scaling, intercept treatment, stopping rules, and score scale differ.

## 6. Fakeddit bounded text experiment

The read-only local TSV check verified raw `1 = SOURCE_TRUE` through six-way class 0 and raw `0 = SOURCE_NON_TRUE` through six-way classes 1–5. It found 23,320 true and 36,022 non-true source rows, reservoir-sampled 600 rows, fit 300 train-only TF-IDF features on 475 rows, and evaluated 125 held-out demonstration rows.

The custom primal model reached accuracy `0.616`, macro-F1 `0.615`, weighted-F1 `0.615`, and balanced accuracy `0.616`; its matrix was `[[42,20],[28,35]]` ordered `[SOURCE_NON_TRUE, SOURCE_TRUE]`. This result is not comparable with the project baseline and is not a factuality claim.

## 7. Conclusions and future work

The tests and small experiments support the mechanics of the implementation. Kernel methods solve the nonlinear XOR example but carry quadratic Gram-matrix storage; the linear primal route is the practical direction for high-dimensional TF-IDF. Future work should add sparse linear updates, an independently frozen validation protocol, stronger SMO working-set selection, and repeated datasets rather than repeated seeds on one test set.

See [limitations](LIMITATIONS.md), [benchmark notes](BENCHMARK_REPORT.md), and [reproducibility instructions](REPRODUCIBILITY.md). Figures are in [figures](figures/).
