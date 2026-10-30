"""Small public binary SVC interface built on the independent optimizers."""
from .svm_dual import DualSMO
from .svm_primal import PrimalLinearSVM


class SVC:
    """Binary classifier with uncalibrated signed-margin decision scores."""

    def __init__(self, C=1.0, kernel="linear", gamma=1.0, degree=3, coef0=1.0,
                 optimizer="smo", max_iter=1000, random_state=42, learning_rate=0.01,
                 tolerance=1e-5):
        self.C, self.kernel, self.gamma, self.degree, self.coef0 = C, kernel, gamma, degree, coef0
        self.optimizer, self.max_iter, self.random_state = optimizer, max_iter, random_state
        self.learning_rate, self.tolerance = learning_rate, tolerance
        self.classes_, self.model_, self.diagnostics_ = [], None, {}

    def fit(self, X, y):
        classes = sorted(set(y))
        if len(classes) != 2:
            raise ValueError("this educational SVC supports exactly two original labels")
        self.classes_ = classes
        signed = [-1 if label == classes[0] else 1 for label in y]
        if self.optimizer == "primal":
            if self.kernel != "linear":
                raise ValueError("primal optimization is implemented only for a linear kernel")
            self.model_ = PrimalLinearSVM(self.C, self.learning_rate, self.max_iter,
                                          self.random_state, self.tolerance).fit(X, signed)
            self.diagnostics_ = {
                "optimizer": "primal", "epochs": len(self.model_.objective_history),
                "objective_history": self.model_.objective_history,
                "history": self.model_.history_, "stopping_reason": self.model_.stopping_reason_,
                "margin_violations": self.model_.margin_violations(X, signed),
                "objective_change_heuristic_triggered": self.model_.converged_,
            }
        elif self.optimizer == "smo":
            self.model_ = DualSMO(self.C, self.kernel, self.gamma, self.degree, self.coef0,
                                  tolerance=max(self.tolerance, 1e-4), max_iter=self.max_iter,
                                  random_state=self.random_state).fit(X, signed)
            self.diagnostics_ = {
                "optimizer": "smo", "iterations": self.model_.iterations_,
                "support_vectors": len(self.model_.support_indices()),
                "equality_residual": self.model_.equality_residual(),
                "kkt_violations": self.model_.kkt_violations(), "max_kkt_residual": self.model_.max_kkt_residual(),
                "max_box_constraint_violation": self.model_.max_box_constraint_violation(),
                "alpha_updates": self.model_.alpha_updates_, "history": self.model_.history_,
                "stopping_reason": self.model_.stopping_reason_, "converged": self.model_.converged_,
            }
        else:
            raise ValueError("optimizer must be 'primal' or 'smo'")
        return self

    def decision_function(self, X):
        if self.model_ is None:
            raise RuntimeError("fit must be called before prediction")
        return self.model_.decision_function(X)

    def predict(self, X):
        return [self.classes_[1] if score >= 0.0 else self.classes_[0]
                for score in self.decision_function(X)]

    @property
    def support_indices_(self):
        if isinstance(self.model_, DualSMO):
            return self.model_.support_indices()
        return []
