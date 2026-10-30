"""Execute the bounded, reproducible research suite without third-party ML code."""
import csv
import json
import platform
import random
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

from src.metrics import classification_report
from src.svc import SVC

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 20261027


def split_rows(X, y, train_ratio=0.5, validation_ratio=0.25):
    rows = list(zip(X, y)); random.Random(SEED).shuffle(rows)
    train_end, validation_end = int(len(rows) * train_ratio), int(len(rows) * (train_ratio + validation_ratio))
    return rows[:train_end], rows[train_end:validation_end], rows[validation_end:]


def gaussian_data(count_per_class=50, overlap=0.0, imbalance=1.0, dimensions=2, seed=SEED):
    rng = random.Random(seed); X, y = [], []
    negatives, positives = count_per_class, max(1, int(count_per_class * imbalance))
    for label, count, center in ((0, negatives, -1.0 + overlap), (1, positives, 1.0 - overlap)):
        for _ in range(count):
            X.append([rng.gauss(center, 0.65) for _ in range(dimensions)]); y.append(label)
    return X, y


def xor_rows(repetitions=20, seed=SEED):
    rng = random.Random(seed); X, y = [], []
    for _ in range(repetitions):
        for point, label in [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]:
            X.append([value + rng.gauss(0, 0.06) for value in point]); y.append(label)
    return X, y


def run_model(identifier, dataset, train, validation, test, config):
    X_train, y_train = zip(*train); X_validation, y_validation = zip(*validation); X_test, y_test = zip(*test)
    tracemalloc.start(); started = time.perf_counter()
    model = SVC(random_state=SEED, **config).fit(list(X_train), list(y_train))
    training_seconds = time.perf_counter() - started; _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    prediction_started = time.perf_counter(); predictions = model.predict(list(X_test)); prediction_seconds = time.perf_counter() - prediction_started
    return {
        "experiment_id": identifier, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "completed",
        "dataset": dataset, "seed": SEED, "train_size": len(train), "validation_size": len(validation), "test_size": len(test),
        "features": len(X_train[0]), "config": config, "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds, "peak_tracemalloc_bytes": peak,
        "validation_metrics": classification_report(list(y_validation), model.predict(list(X_validation)), labels=[0, 1]),
        "test_metrics": classification_report(list(y_test), predictions, labels=[0, 1]),
        "test_predictions": [{"truth": truth, "prediction": prediction, "score": score}
                             for truth, prediction, score in zip(y_test, predictions, model.decision_function(list(X_test)))],
        "diagnostics": model.diagnostics_,
    }


def main():
    for directory in (RESULTS / "raw", RESULTS / "tables", RESULTS / "experiments", RESULTS / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    # Configurations are chosen on validation only; each test set is read once per fixed run.
    separable = split_rows(*gaussian_data(overlap=0.0, seed=11))
    overlap = split_rows(*gaussian_data(overlap=0.75, seed=12))
    xor = split_rows(*xor_rows())
    imbalanced = split_rows(*gaussian_data(count_per_class=40, overlap=0.55, imbalance=0.2, seed=13))
    runs = []
    runs.append(run_model("A-primal-linear", "separable-gaussian", *separable, {"optimizer": "primal", "kernel": "linear", "C": 1.0, "learning_rate": 0.003, "max_iter": 400}))
    runs.append(run_model("A-dual-linear", "separable-gaussian", *separable, {"optimizer": "smo", "kernel": "linear", "C": 1.0, "max_iter": 3000}))
    for kernel, extra in (("linear", {}), ("poly", {"degree": 2, "gamma": 1.0}), ("rbf", {"gamma": 2.0})):
        runs.append(run_model(f"C-xor-{kernel}", "xor-noisy", *xor, {"optimizer": "smo", "kernel": kernel, "C": 10.0, "max_iter": 5000, **extra}))
    runs.append(run_model("D-imbalanced-linear", "imbalanced-overlap", *imbalanced, {"optimizer": "primal", "kernel": "linear", "C": 1.0, "learning_rate": 0.002, "max_iter": 400}))
    # Validation-only sweeps on the overlap dataset; test values retained but never selected upon.
    for C in (0.01, 0.1, 1.0, 10.0, 100.0):
        runs.append(run_model(f"B-C-{C}", "overlap-gaussian", *overlap, {"optimizer": "primal", "kernel": "linear", "C": C, "learning_rate": 0.001, "max_iter": 400}))
    for gamma in (0.001, 0.01, 0.1, 1.0, 10.0):
        runs.append(run_model(f"C-gamma-{gamma}", "xor-noisy", *xor, {"optimizer": "smo", "kernel": "rbf", "C": 10.0, "gamma": gamma, "max_iter": 5000}))
    # Empirical complexity runs remain intentionally small because the kernel Gram matrix is quadratic.
    for samples in (40, 80, 120):
        X, y = gaussian_data(count_per_class=samples // 2, overlap=0.5, dimensions=10, seed=samples)
        train, validation, test = split_rows(X, y)
        runs.append(run_model(f"scale-primal-{samples}", f"scaling-{samples}x10", train, validation, test, {"optimizer": "primal", "kernel": "linear", "C": 1.0, "learning_rate": 0.001, "max_iter": 150}))
        runs.append(run_model(f"scale-dual-{samples}", f"scaling-{samples}x10", train, validation, test, {"optimizer": "smo", "kernel": "rbf", "C": 1.0, "gamma": 0.1, "max_iter": 1000}))
    payload = {"suite_id": "svc-scratch-research-v1", "generated_utc": datetime.now(timezone.utc).isoformat(), "environment": {"python": sys.version, "platform": platform.platform()}, "runs": runs}
    raw_path = RESULTS / "raw" / "svc-scratch-research-v1.json"; raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    columns = ["experiment_id", "dataset", "optimizer", "kernel", "C", "gamma", "test_accuracy", "test_macro_f1", "test_weighted_f1", "test_balanced_accuracy", "training_seconds", "prediction_seconds", "peak_tracemalloc_bytes", "status"]
    with (RESULTS / "tables" / "experiment_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        for run in runs:
            metrics, config = run["test_metrics"], run["config"]
            writer.writerow({"experiment_id": run["experiment_id"], "dataset": run["dataset"], "optimizer": config["optimizer"], "kernel": config["kernel"], "C": config["C"], "gamma": config.get("gamma", ""), "test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"], "test_weighted_f1": metrics["weighted_f1"], "test_balanced_accuracy": metrics["balanced_accuracy"], "training_seconds": run["training_seconds"], "prediction_seconds": run["prediction_seconds"], "peak_tracemalloc_bytes": run["peak_tracemalloc_bytes"], "status": run["status"]})
    print(raw_path)


if __name__ == "__main__":
    main()
