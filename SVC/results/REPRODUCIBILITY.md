# Reproducibility

Run from `SVC_FROM_SCRATCH`:

```bash
python3 run_tests.py
python3 run_research_suite.py
uv run python render_results.py
uv run python reference_comparison.py
```

The first two commands use standard-library SVC code. Matplotlib is used only by the reporting layer and scikit-learn only by the optional reference script. The suite seed is `20261027`; raw logs capture timestamp, platform, Python version, split sizes, parameters, metrics, and optimization traces.
