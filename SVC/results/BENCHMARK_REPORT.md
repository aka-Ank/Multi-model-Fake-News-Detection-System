# Benchmark report

The bounded scaling runs use 10 features and 40, 80, and 120 total synthetic observations. The primal solver stores O(d) parameters and processes O(nd) work per epoch. Kernel SMO materializes an O(n²) Gram matrix; its wall time and memory are therefore data- and working-set-dependent and should not be extrapolated beyond these small runs.

The suite records `tracemalloc` peak allocation, training and prediction durations, support-vector counts, iteration histories, and stopping reasons in the raw JSON and [summary table](tables/experiment_summary.csv). Runs were intentionally capped before a large-scale kernel benchmark could exhaust resources.
