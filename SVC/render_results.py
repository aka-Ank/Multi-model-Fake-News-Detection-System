"""Optional matplotlib reporting layer; never imported by the SVC core."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"


def save(name, title, xlabel, ylabel):
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / name, dpi=180); plt.close()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = json.loads((RESULTS / "raw" / "svc-scratch-research-v1.json").read_text())["runs"]
    by_id = {run["experiment_id"]: run for run in runs}
    primal = by_id["A-primal-linear"]["diagnostics"]["history"]
    epoch = [row["epoch"] for row in primal]
    for key, title, ylabel, filename in [("objective", "Primal objective convergence", "objective", "01_primal_objective.png"), ("hinge_loss_sum", "Primal hinge-loss sum", "summed hinge loss", "02_primal_hinge.png"), ("regularization", "Primal regularization term", "0.5 ||w||²", "03_primal_regularization.png"), ("weight_norm", "Primal weight norm", "||w||", "04_primal_weight_norm.png")]:
        plt.plot(epoch, [row[key] for row in primal], label="linear primal")
        save(filename, title + " (separable Gaussian; C=1)", "epoch", ylabel)
    dual = by_id["A-dual-linear"]["diagnostics"]["history"]
    iteration = [row["iteration"] for row in dual]
    for key, title, ylabel, filename in [("dual_objective", "Dual objective convergence", "dual objective", "05_dual_objective.png"), ("max_kkt_residual", "SMO KKT residual", "maximum residual", "06_dual_kkt.png")]:
        plt.plot(iteration, [row[key] for row in dual], label="linear SMO")
        save(filename, title + " (separable Gaussian; C=1)", "SMO pass", ylabel)
    c_runs = [run for run in runs if run["experiment_id"].startswith("B-C-")]
    c = [run["config"]["C"] for run in c_runs]
    for field, title, filename in [("accuracy", "Validation accuracy versus C", "07_accuracy_vs_c.png"), ("macro_f1", "Validation macro-F1 versus C", "08_macro_f1_vs_c.png")]:
        plt.semilogx(c, [run["validation_metrics"][field] for run in c_runs], marker="o", label="overlap validation")
        save(filename, title + " (linear primal)", "C", field.replace("_", " "))
    gamma_runs = [run for run in runs if run["experiment_id"].startswith("C-gamma-")]
    gamma = [run["config"]["gamma"] for run in gamma_runs]
    plt.semilogx(gamma, [run["validation_metrics"]["accuracy"] for run in gamma_runs], marker="o", label="XOR validation")
    save("09_accuracy_vs_gamma.png", "Validation accuracy versus RBF gamma (XOR; C=10)", "gamma", "accuracy")
    xor = [run for run in runs if run["experiment_id"].startswith("C-xor-")]
    plt.bar([run["config"]["kernel"] for run in xor], [run["test_metrics"]["macro_f1"] for run in xor], label="held-out XOR macro-F1")
    save("10_kernel_comparison.png", "Kernel comparison on held-out noisy XOR", "kernel", "macro-F1")
    pair = [by_id["A-primal-linear"], by_id["A-dual-linear"]]
    plt.bar([run["config"]["optimizer"] for run in pair], [run["test_metrics"]["macro_f1"] for run in pair], label="held-out macro-F1")
    save("11_primal_dual_comparison.png", "Primal versus dual on held-out separable data", "optimizer", "macro-F1")
    scale = [run for run in runs if run["experiment_id"].startswith("scale-")]
    for optimizer in ("primal", "smo"):
        rows = [run for run in scale if run["config"]["optimizer"] == optimizer]
        plt.plot([run["train_size"] for run in rows], [run["training_seconds"] for run in rows], marker="o", label=optimizer)
    save("12_training_scaling.png", "Training-time scaling (10 features)", "training samples", "seconds")
    imbalance = by_id["D-imbalanced-linear"]["test_metrics"]
    matrix = imbalance["confusion_matrix"]
    plt.imshow(matrix, cmap="Blues"); plt.colorbar(label="test records")
    for row in range(2):
        for col in range(2): plt.text(col, row, str(matrix[row][col]), ha="center", va="center")
    plt.xticks([0, 1], ["non-true", "true"]); plt.yticks([0, 1], ["non-true", "true"])
    plt.title("Imbalanced held-out confusion matrix (linear primal)"); plt.xlabel("predicted class"); plt.ylabel("true class"); plt.tight_layout()
    plt.savefig(FIG / "13_imbalance_confusion_matrix.png", dpi=180); plt.close()


if __name__ == "__main__":
    main()
