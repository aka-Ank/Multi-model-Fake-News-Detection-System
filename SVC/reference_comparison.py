"""Optional external comparison. This file is deliberately outside the core SVC package."""
import json
import time
from pathlib import Path

from sklearn.svm import LinearSVC, SVC as SklearnSVC

from run_research_suite import SEED, gaussian_data, split_rows
from src.metrics import classification_report
from src.svc import SVC


def timed_model(name, estimator, X_train, y_train, X_test, y_test):
    started = time.perf_counter(); estimator.fit(X_train, y_train); train_seconds = time.perf_counter() - started
    started = time.perf_counter(); prediction = list(estimator.predict(X_test)); predict_seconds = time.perf_counter() - started
    return {"name": name, "training_seconds": train_seconds, "prediction_seconds": predict_seconds,
            "metrics": classification_report(y_test, prediction, labels=[0, 1]), "predictions": prediction}


def main():
    X, y = gaussian_data(count_per_class=50, overlap=0.5, seed=91)
    train, _validation, test = split_rows(X, y)
    X_train, y_train = map(list, zip(*train)); X_test, y_test = map(list, zip(*test))
    models = [
        ("custom_primal_linear", SVC(C=1.0, optimizer="primal", kernel="linear", learning_rate=0.002, max_iter=400, random_state=SEED)),
        ("custom_dual_linear", SVC(C=1.0, optimizer="smo", kernel="linear", max_iter=3000, random_state=SEED)),
        ("custom_rbf", SVC(C=1.0, optimizer="smo", kernel="rbf", gamma=0.5, max_iter=3000, random_state=SEED)),
        ("sklearn_linear_svc", LinearSVC(C=1.0, max_iter=3000, random_state=SEED)),
        ("sklearn_rbf_svc", SklearnSVC(C=1.0, kernel="rbf", gamma=0.5, random_state=SEED)),
    ]
    records = [timed_model(name, estimator, X_train, y_train, X_test, y_test) for name, estimator in models]
    baseline = records[0]["predictions"]
    for record in records:
        record["agreement_with_custom_primal"] = sum(a == b for a, b in zip(baseline, record["predictions"])) / len(baseline)
        del record["predictions"]
    output = {"dataset": "gaussian-overlap-reference", "seed": SEED, "train_size": len(train), "test_size": len(test), "records": records,
              "note": "Agreement does not require matching parameters or scores: loss scaling, intercept treatment and stopping criteria differ."}
    path = Path("results/raw/reference_comparison.json"); path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
