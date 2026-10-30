"""Linear soft-margin SVM trained with a mini-batch subgradient method."""
import random

from .linalg import dot, validate_matrix


class PrimalLinearSVM:
    """Optimizes .5||w||^2 + C sum_i max(0, 1-y_i(w.x_i+b)).

    At exactly margin one, this implementation selects the zero hinge-loss
    subgradient. Batch gradients are scaled to be unbiased estimates of the
    full summed-hinge objective.
    """

    def __init__(self, C=1.0, learning_rate=0.01, epochs=100, random_state=42,
                 tolerance=1e-8, batch_size=32):
        if C <= 0 or learning_rate <= 0 or epochs < 1 or batch_size < 1:
            raise ValueError("C, learning_rate, epochs, and batch_size must be positive")
        self.C, self.learning_rate, self.epochs = float(C), float(learning_rate), int(epochs)
        self.random_state, self.tolerance, self.batch_size = random_state, tolerance, int(batch_size)
        self.weights, self.bias, self.objective_history, self.history_, self.converged_ = [], 0.0, [], [], False
        self.stopping_reason_ = "not_started"

    def _validate(self, X, y):
        validate_matrix(X, "X")
        if len(X) != len(y) or not set(y).issubset({-1, 1}):
            raise ValueError("X and signed y in {-1,+1} must have matching lengths")

    def decision_function(self, X):
        if not self.weights:
            raise RuntimeError("fit must be called before prediction")
        return [dot(self.weights, row) + self.bias for row in X]

    def objective(self, X, y):
        scores = self.decision_function(X)
        return 0.5 * dot(self.weights, self.weights) + self.C * sum(
            max(0.0, 1.0 - label * score) for label, score in zip(y, scores)
        )

    def objective_terms(self, X, y):
        scores = self.decision_function(X)
        hinge = sum(max(0.0, 1.0 - label * score) for label, score in zip(y, scores))
        regularization = 0.5 * dot(self.weights, self.weights)
        return hinge, regularization, regularization + self.C * hinge

    def gradient(self, X, y):
        """Full objective subgradient, useful for finite-difference tests."""
        grad_w, grad_b = list(self.weights), 0.0
        for row, label, score in zip(X, y, self.decision_function(X)):
            if label * score < 1.0:
                for index, value in enumerate(row):
                    grad_w[index] -= self.C * label * value
                grad_b -= self.C * label
        return grad_w, grad_b

    def fit(self, X, y):
        self._validate(X, y)
        n, width = len(X), len(X[0])
        self.weights, self.bias, self.objective_history, self.history_, self.converged_ = [0.0] * width, 0.0, [], [], False
        self.stopping_reason_ = "max_epochs"
        order, rng, previous = list(range(n)), random.Random(self.random_state), None
        for epoch in range(self.epochs):
            rng.shuffle(order)
            rate = self.learning_rate / (1.0 + 0.02 * epoch)
            for begin in range(0, n, self.batch_size):
                indices = order[begin:begin + self.batch_size]
                # n/batch scales this sample gradient to the exact summed loss convention.
                scale = n / len(indices)
                grad_w, grad_b = list(self.weights), 0.0
                for index in indices:
                    margin = y[index] * (dot(self.weights, X[index]) + self.bias)
                    if margin < 1.0:
                        for feature, value in enumerate(X[index]):
                            grad_w[feature] -= self.C * scale * y[index] * value
                        grad_b -= self.C * scale * y[index]
                self.weights = [weight - rate * grad for weight, grad in zip(self.weights, grad_w)]
                self.bias -= rate * grad_b
            hinge, regularization, current = self.objective_terms(X, y)
            self.objective_history.append(current)
            self.history_.append({"epoch": epoch + 1, "learning_rate": rate, "hinge_loss_sum": hinge,
                                  "regularization": regularization, "objective": current,
                                  "weight_norm": dot(self.weights, self.weights) ** 0.5,
                                  "margin_violations": self.margin_violations(X, y)})
            if previous is not None and abs(previous - current) <= self.tolerance * max(1.0, abs(previous)):
                self.converged_ = True
                self.stopping_reason_ = "objective_change_heuristic"
                break
            previous = current
        return self

    def predict_signed(self, X):
        return [1 if score >= 0.0 else -1 for score in self.decision_function(X)]

    def margin_violations(self, X, y):
        return sum(label * score < 1.0 for label, score in zip(y, self.decision_function(X)))
