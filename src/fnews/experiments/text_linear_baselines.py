from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fnews.experiments.text_logreg_baseline import load_config, load_fakeddit_split
from fnews.ml.features.tfidf import ManualTfidfVectorizer
from fnews.ml.metrics.classification import binary_classification_metrics, confusion_matrix_binary
from fnews.ml.models.linear_svm import ManualLinearSVM
from fnews.ml.models.logistic_regression import ManualLogisticRegression, sigmoid_stable
from fnews.ml.preprocessing.split import stratified_sample_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manual text linear baselines across seeds.")
    parser.add_argument("--config", default="configs/experiments/text_linear_50k_multiseed.json")
    return parser.parse_args()


def run_experiment(config: dict[str, object]) -> Path:
    started = time.perf_counter()
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    train_frame = load_fakeddit_split(str(config["train_tsv"]))
    validation_frame = load_fakeddit_split(str(config["validation_tsv"]))
    validation_labels = validation_frame["2_way_label"].to_numpy(dtype=int)

    all_runs: list[dict[str, object]] = []
    for seed in [int(seed) for seed in config["seeds"]]:
        sample_indices = stratified_sample_indices(
            train_frame["2_way_label"].to_numpy(),
            int(config["sample_size"]),
            seed,
        )
        sampled_train = train_frame.iloc[sample_indices].reset_index(drop=True)
        train_labels = sampled_train["2_way_label"].to_numpy(dtype=int)

        vectorizer = ManualTfidfVectorizer(
            min_frequency=int(config["min_frequency"]),
            max_vocabulary_size=int(config["max_vocabulary_size"]),
            tokenization_mode=str(config["tokenization_mode"]),
            l2_normalize=bool(config["l2_normalize"]),
        )
        train_vectors = vectorizer.fit_transform(sampled_train["clean_title"].tolist())
        validation_vectors = vectorizer.transform(validation_frame["clean_title"].tolist())

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        sampled_train.to_parquet(seed_dir / "train_subset.parquet", index=False)
        vectorizer.save(seed_dir / "vectorizer")

        all_runs.append(
            train_logistic_regression(
                config,
                seed,
                seed_dir,
                train_vectors,
                train_labels,
                validation_vectors,
                validation_labels,
                vectorizer.n_features,
            )
        )
        all_runs.append(
            train_linear_svm(
                config,
                seed,
                seed_dir,
                train_vectors,
                train_labels,
                validation_vectors,
                validation_labels,
                vectorizer.n_features,
            )
        )

    run_frame = flatten_run_metrics(all_runs)
    run_frame.to_csv(output_dir / "run_metrics.csv", index=False)
    aggregate = aggregate_runs(run_frame)
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)

    metadata = {
        "config": config,
        "runtime_seconds": time.perf_counter() - started,
        "runs": all_runs,
        "aggregate": aggregate.to_dict(orient="records"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    Path(str(config["config_copy_path"])).parent.mkdir(parents=True, exist_ok=True)
    Path(str(config["config_copy_path"])).write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(output_dir / "summary.md", metadata, run_frame, aggregate)
    return output_dir


def train_logistic_regression(
    config: dict[str, object],
    seed: int,
    seed_dir: Path,
    train_vectors: list[dict[int, float]],
    train_labels: np.ndarray,
    validation_vectors: list[dict[int, float]],
    validation_labels: np.ndarray,
    n_features: int,
) -> dict[str, object]:
    run_dir = seed_dir / "logistic_regression"
    run_dir.mkdir(exist_ok=True)
    model = ManualLogisticRegression(
        n_features=n_features,
        learning_rate=float(config["logistic_regression"]["learning_rate"]),
        l2=float(config["logistic_regression"]["l2"]),
        seed=seed,
    )
    history = model.fit(
        train_vectors,
        train_labels,
        validation_vectors=validation_vectors,
        validation_labels=validation_labels,
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        gradient_clip=float(config["gradient_clip"]),
        early_stopping_patience=int(config["early_stopping_patience"]),
    )
    scores = model.predict_proba(validation_vectors)
    predictions = (scores >= float(config["threshold"])).astype(int)
    return persist_run(
        "logistic_regression",
        seed,
        run_dir,
        model,
        history,
        validation_labels,
        scores,
        predictions,
    )


def train_linear_svm(
    config: dict[str, object],
    seed: int,
    seed_dir: Path,
    train_vectors: list[dict[int, float]],
    train_labels: np.ndarray,
    validation_vectors: list[dict[int, float]],
    validation_labels: np.ndarray,
    n_features: int,
) -> dict[str, object]:
    run_dir = seed_dir / "linear_svm"
    run_dir.mkdir(exist_ok=True)
    model = ManualLinearSVM(
        n_features=n_features,
        learning_rate=float(config["linear_svm"]["learning_rate"]),
        l2=float(config["linear_svm"]["l2"]),
        seed=seed,
    )
    history = model.fit(
        train_vectors,
        train_labels,
        validation_vectors=validation_vectors,
        validation_labels=validation_labels,
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        gradient_clip=float(config["gradient_clip"]),
        early_stopping_patience=int(config["early_stopping_patience"]),
    )
    decision_scores = model.decision_function(validation_vectors)
    predictions = (decision_scores >= 0).astype(int)
    pseudo_probabilities = sigmoid_stable(decision_scores)
    return persist_run(
        "linear_svm",
        seed,
        run_dir,
        model,
        history,
        validation_labels,
        pseudo_probabilities,
        predictions,
        raw_scores=decision_scores,
    )


def persist_run(
    model_name: str,
    seed: int,
    run_dir: Path,
    model: ManualLogisticRegression | ManualLinearSVM,
    history: list[dict[str, float]],
    validation_labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    raw_scores: np.ndarray | None = None,
) -> dict[str, object]:
    metrics = binary_classification_metrics(validation_labels, predictions, scores)
    confusion = confusion_matrix_binary(validation_labels, predictions)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame(confusion, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        run_dir / "confusion_matrix.csv"
    )
    prediction_frame = pd.DataFrame(
        {
            "label": validation_labels,
            "score_fake": scores,
            "prediction": predictions,
        }
    )
    if raw_scores is not None:
        prediction_frame["raw_decision_score"] = raw_scores
    prediction_frame.to_parquet(run_dir / "predictions.parquet", index=False)
    model.save(
        run_dir / "model",
        metadata={
            "model_name": model_name,
            "label_mapping": {"0": "true", "1": "fake"},
        },
    )
    plot_history(pd.DataFrame(history), run_dir / "training_curves.png", model_name)
    model_size = int((run_dir / "model" / "model.npz").stat().st_size)
    payload = {
        "model": model_name,
        "seed": seed,
        "metrics": metrics,
        "confusion_matrix": confusion.tolist(),
        "epochs_ran": len(history),
        "model_size_bytes": model_size,
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def plot_history(history: pd.DataFrame, path: Path, title: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["train_loss"], label="train loss")
    if "validation_loss" in history:
        plt.plot(history["epoch"], history["validation_loss"], label="validation loss")
    plt.title(title.replace("_", " ").title())
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def flatten_run_metrics(runs: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        metrics = run["metrics"]
        assert isinstance(metrics, dict)
        row = {
            "model": run["model"],
            "seed": run["seed"],
            "epochs_ran": run["epochs_ran"],
            "model_size_bytes": run["model_size_bytes"],
        }
        for key, value in metrics.items():
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_runs(run_frame: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "weighted_f1",
        "roc_auc",
        "brier_score",
    ]
    rows = []
    for model_name, group in run_frame.groupby("model"):
        row: dict[str, object] = {"model": model_name, "n_runs": len(group)}
        for metric in metric_names:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model")


def write_summary(
    path: Path,
    metadata: dict[str, object],
    run_frame: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> None:
    lines = [
        "# Manual Text Linear Baselines: 50k Multi-Seed",
        "",
        f"- Runtime seconds: {metadata['runtime_seconds']:.2f}",
        f"- Seeds: {', '.join(str(seed) for seed in metadata['config']['seeds'])}",
        f"- Train subset per seed: {metadata['config']['sample_size']:,}",
        "- Validation split: official multimodal validation",
        "- Official test split: not used",
        "",
        "## Aggregate Metrics",
        "",
        aggregate.to_markdown(index=False),
        "",
        "## Per-Run Metrics",
        "",
        run_frame[
            [
                "model",
                "seed",
                "accuracy",
                "precision",
                "recall",
                "macro_f1",
                "weighted_f1",
                "roc_auc",
                "brier_score",
            ]
        ].to_markdown(index=False),
        "",
        "Note: SVM Brier score uses a sigmoid transform of the raw margin, so treat it as a rough calibration diagnostic rather than a calibrated probability score.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = run_experiment(load_config(args.config))
    print(f"Wrote experiment outputs under {output_dir}")


if __name__ == "__main__":
    main()
