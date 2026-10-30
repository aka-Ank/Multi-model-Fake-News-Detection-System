"""Reproducible synthetic and optional TSV experiments for the scratch SVC."""
import csv
import json
import random
import time
from pathlib import Path

from src.metrics import classification_report
from src.svc import SVC
from src.text_vectorizer import TfidfVectorizer, memory_estimate


def xor_data():
    return [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [0, 1, 1, 0]


def separable_data():
    return [[-2.0, -1.0], [-1.0, -2.0], [-1.5, -1.0], [1.0, 1.5], [2.0, 1.0], [1.5, 2.0]], [0, 0, 0, 1, 1, 1]


def run_synthetic():
    X, y = separable_data()
    results = {}
    for name, kwargs in {
        "primal_linear": {"optimizer": "primal", "kernel": "linear", "learning_rate": 0.005, "max_iter": 500},
        "dual_linear": {"optimizer": "smo", "kernel": "linear", "max_iter": 3000},
    }.items():
        model = SVC(C=1.0, random_state=7, **kwargs).fit(X, y)
        results[name] = {"report": classification_report(y, model.predict(X), labels=[0, 1]), "diagnostics": model.diagnostics_}
    X_xor, y_xor = xor_data()
    for name, kwargs in {
        "xor_linear": {"optimizer": "smo", "kernel": "linear"},
        "xor_rbf": {"optimizer": "smo", "kernel": "rbf", "gamma": 2.0},
    }.items():
        model = SVC(C=10.0, max_iter=5000, random_state=7, **kwargs).fit(X_xor, y_xor)
        results[name] = {"report": classification_report(y_xor, model.predict(X_xor), labels=[0, 1]), "diagnostics": model.diagnostics_}
    # Small deterministic sweeps illustrate trade-offs; they are not model selection.
    overlap_X = [[-1.0, -1.0], [-0.3, -0.2], [0.1, 0.2], [0.5, 0.4], [1.0, 1.0]]
    overlap_y = [0, 0, 1, 1, 1]
    c_sweep = []
    for C in (0.1, 1.0, 10.0):
        model = SVC(C=C, optimizer="primal", kernel="linear", learning_rate=0.003, max_iter=300, random_state=7).fit(overlap_X, overlap_y)
        c_sweep.append({"C": C, "accuracy": classification_report(overlap_y, model.predict(overlap_X), [0, 1])["accuracy"], "margin_violations": model.diagnostics_["margin_violations"], "final_objective": model.diagnostics_["objective_history"][-1]})
    gamma_sweep = []
    for gamma in (0.1, 0.5, 2.0):
        model = SVC(C=10.0, optimizer="smo", kernel="rbf", gamma=gamma, max_iter=5000, random_state=7).fit(X_xor, y_xor)
        gamma_sweep.append({"gamma": gamma, "accuracy": classification_report(y_xor, model.predict(X_xor), [0, 1])["accuracy"], "support_vectors": len(model.support_indices_)})
    learning_rate_sweep = []
    for rate in (0.001, 0.005, 0.02):
        model = SVC(C=1.0, optimizer="primal", kernel="linear", learning_rate=rate, max_iter=300, random_state=7).fit(X, y)
        learning_rate_sweep.append({"learning_rate": rate, "epochs": len(model.diagnostics_["objective_history"]), "final_objective": model.diagnostics_["objective_history"][-1], "accuracy": classification_report(y, model.predict(X), [0, 1])["accuracy"]})
    iteration_sweep = []
    for epochs in (10, 50, 200):
        model = SVC(C=1.0, optimizer="primal", kernel="linear", learning_rate=0.005, max_iter=epochs, random_state=7).fit(X, y)
        iteration_sweep.append({"epochs_requested": epochs, "epochs_run": len(model.diagnostics_["objective_history"]), "final_objective": model.diagnostics_["objective_history"][-1], "accuracy": classification_report(y, model.predict(X), [0, 1])["accuracy"]})
    results["sensitivity"] = {"C": c_sweep, "gamma": gamma_sweep, "learning_rate": learning_rate_sweep, "iterations": iteration_sweep, "dual_gram_bytes_estimate": len(X) * len(X) * 8, "dense_feature_bytes_estimate": memory_estimate(len(X), len(X[0]))}
    return results


def optional_tsv_experiment(path, limit=400, max_features=300):
    """Bounded experiment for a user-selected TSV with `text` and `label` fields.

    It intentionally requires explicit column names from the caller. This avoids
    silently applying Fakeddit's original release labels as if they were the
    FNEWS research-dataset source taxonomy.
    """
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("text", "").strip() and row.get("label", "") in {"0", "1"}:
                rows.append((row["text"], int(row["label"])))
            if len(rows) >= limit:
                break
    if len(set(label for _, label in rows)) != 2:
        raise ValueError("selected TSV must supply both labels in text and label columns")
    random.Random(42).shuffle(rows)
    cut = max(1, int(len(rows) * 0.75)); train, test = rows[:cut], rows[cut:]
    vectorizer = TfidfVectorizer(max_features=max_features).fit([text for text, _ in train])
    X_train, X_test = vectorizer.transform([text for text, _ in train]), vectorizer.transform([text for text, _ in test])
    model = SVC(C=1.0, kernel="linear", optimizer="primal", learning_rate=0.002, max_iter=200, random_state=42).fit(X_train, [label for _, label in train])
    return {"rows": len(rows), "vocabulary": len(vectorizer.vocabulary_), "dense_bytes_estimate": memory_estimate(len(rows), len(vectorizer.vocabulary_)), "report": classification_report([label for _, label in test], model.predict(X_test), labels=[0, 1]), "diagnostics": model.diagnostics_}


def fnews_fakeddit_tsv_experiment(path, per_class=300, max_features=300, random_state=42):
    """Run a read-only bounded Fakeddit experiment after checking taxonomy semantics.

    The local official TSV has `clean_title`, `2_way_label`, and `6_way_label`.
    We verify that raw 1 joins only to six-way 0 and raw 0 joins only to 1..5
    before using its labels. Reservoir sampling visits the complete supplied TSV
    without retaining the full dataset; vectorizer fitting uses train text only.
    """
    reservoirs, seen, rng = {0: [], 1: []}, {0: 0, 1: 0}, random.Random(random_state)
    mapping_errors = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                label, six_way = int(row["2_way_label"]), int(row["6_way_label"])
            except (KeyError, TypeError, ValueError):
                continue
            valid_mapping = (label == 1 and six_way == 0) or (label == 0 and six_way in {1, 2, 3, 4, 5})
            if not valid_mapping:
                mapping_errors.append((row.get("id", ""), label, six_way))
                continue
            text = row.get("clean_title", "").strip()
            if not text:
                continue
            seen[label] += 1
            entry = (row.get("id", ""), text, label)
            if len(reservoirs[label]) < per_class:
                reservoirs[label].append(entry)
            else:
                replacement = rng.randrange(seen[label])
                if replacement < per_class:
                    reservoirs[label][replacement] = entry
    if mapping_errors or not all(reservoirs.values()):
        raise ValueError(f"Fakeddit taxonomy validation failed; invalid rows={len(mapping_errors)}")
    sampled = reservoirs[0] + reservoirs[1]
    sampled.sort(key=lambda row: row[0])
    train, test = [], []
    for identifier, text, label in sampled:
        # Stable split from a simple deterministic character checksum; no test text enters fit.
        (test if sum(map(ord, identifier)) % 5 == 0 else train).append((text, label))
    vectorizer = TfidfVectorizer(max_features=max_features).fit([text for text, _ in train])
    X_train = vectorizer.transform([text for text, _ in train])
    X_test = vectorizer.transform([text for text, _ in test])
    model = SVC(C=1.0, kernel="linear", optimizer="primal", learning_rate=0.002,
                max_iter=200, random_state=random_state).fit(X_train, [label for _, label in train])
    return {
        "source": str(path), "taxonomy_verified": True,
        "mapping": {"1": "SOURCE_TRUE", "0": "SOURCE_NON_TRUE"},
        "source_rows_by_label": seen, "sampled_rows": len(sampled), "train_rows": len(train), "test_rows": len(test),
        "vocabulary": len(vectorizer.vocabulary_), "dense_bytes_estimate": memory_estimate(len(sampled), len(vectorizer.vocabulary_)),
        "report": classification_report([label for _, label in test], model.predict(X_test), labels=[0, 1]),
        "diagnostics": model.diagnostics_,
    }


def main():
    started = time.perf_counter(); results = run_synthetic(); results["runtime_seconds"] = time.perf_counter() - started
    destination = Path("reports"); destination.mkdir(exist_ok=True)
    (destination / "synthetic_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
