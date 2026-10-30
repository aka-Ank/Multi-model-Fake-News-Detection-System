"""Binary kernel SVM using Platt-style sequential minimal optimization."""
import random

from .kernels import resolve
from .linalg import validate_matrix


class DualSMO:
    def __init__(self, C=1.0, kernel="rbf", gamma=1.0, degree=3, coef0=1.0,
                 tolerance=1e-3, epsilon=1e-5, max_iter=10000, max_passes=10,
                 random_state=42):
        if C <= 0 or max_iter < 1 or max_passes < 1:
            raise ValueError("C, max_iter, and max_passes must be positive")
        self.C, self.kernel_name, self.gamma = float(C), kernel, float(gamma)
        self.degree, self.coef0, self.tolerance, self.epsilon = degree, coef0, tolerance, epsilon
        self.max_iter, self.max_passes, self.random_state = max_iter, max_passes, random_state
        self.alphas, self.bias, self.X, self.y, self.gram = [], 0.0, [], [], []
        self.iterations_, self.passes_without_change_, self.converged_ = 0, 0, False
        self.history_, self.alpha_updates_, self.stopping_reason_ = [], 0, "not_started"

    def dual_objective(self):
        linear = sum(self.alphas)
        quadratic = sum(self.alphas[i] * self.y[i] * self.alphas[j] * self.y[j] * self.gram[i][j]
                        for i in range(len(self.alphas)) for j in range(len(self.alphas)))
        return linear - 0.5 * quadratic

    def max_box_constraint_violation(self):
        return max([max(0.0, -alpha, alpha - self.C) for alpha in self.alphas] or [0.0])

    def max_kkt_residual(self):
        residuals = []
        for index, (alpha, label) in enumerate(zip(self.alphas, self.y)):
            margin = label * self._score_index(index)
            if alpha <= self.epsilon:
                residuals.append(max(0.0, 1.0 - margin))
            elif alpha >= self.C - self.epsilon:
                residuals.append(max(0.0, margin - 1.0))
            else:
                residuals.append(abs(margin - 1.0))
        return max(residuals or [0.0])

    def _score_index(self, index):
        return self.bias + sum(alpha * label * self.gram[row][index]
                               for row, (alpha, label) in enumerate(zip(self.alphas, self.y)) if alpha)

    def _error(self, index):
        return self._score_index(index) - self.y[index]

    def _take_step(self, first, second):
        if first == second:
            return False
        a1, a2, y1, y2 = self.alphas[first], self.alphas[second], self.y[first], self.y[second]
        e1, e2 = self._error(first), self._error(second)
        if y1 != y2:
            lower, upper = max(0.0, a2 - a1), min(self.C, self.C + a2 - a1)
        else:
            lower, upper = max(0.0, a1 + a2 - self.C), min(self.C, a1 + a2)
        if lower >= upper - self.epsilon:
            return False
        k11, k12, k22 = self.gram[first][first], self.gram[first][second], self.gram[second][second]
        eta = k11 + k22 - 2.0 * k12
        if eta > self.epsilon:
            new_a2 = a2 + y2 * (e1 - e2) / eta
            new_a2 = min(upper, max(lower, new_a2))
        else:
            # Choose the bound with the larger dual-objective improvement.
            def dual_at(candidate):
                candidate_a1 = a1 + y1 * y2 * (a2 - candidate)
                alpha = list(self.alphas); alpha[first], alpha[second] = candidate_a1, candidate
                return sum(alpha) - 0.5 * sum(
                    alpha[i] * self.y[i] * alpha[j] * self.y[j] * self.gram[i][j]
                    for i in range(len(alpha)) for j in range(len(alpha))
                )
            new_a2 = upper if dual_at(upper) > dual_at(lower) else lower
        if abs(new_a2 - a2) < self.epsilon * (new_a2 + a2 + self.epsilon):
            return False
        new_a1 = a1 + y1 * y2 * (a2 - new_a2)
        b_old = self.bias
        b1 = b_old - e1 - y1 * (new_a1 - a1) * k11 - y2 * (new_a2 - a2) * k12
        b2 = b_old - e2 - y1 * (new_a1 - a1) * k12 - y2 * (new_a2 - a2) * k22
        self.bias = b1 if self.epsilon < new_a1 < self.C - self.epsilon else (b2 if self.epsilon < new_a2 < self.C - self.epsilon else (b1 + b2) / 2.0)
        self.alphas[first], self.alphas[second] = new_a1, new_a2
        return True

    def fit(self, X, y):
        validate_matrix(X, "X")
        if len(X) != len(y) or not set(y).issubset({-1, 1}):
            raise ValueError("X and signed y in {-1,+1} must have matching lengths")
        self.X, self.y, self.alphas, self.bias = [list(row) for row in X], list(y), [0.0] * len(X), 0.0
        kernel = resolve(self.kernel_name, self.gamma, self.degree, self.coef0)
        self.gram = [[kernel(left, right) for right in self.X] for left in self.X]
        rng, passes, iteration = random.Random(self.random_state), 0, 0
        self.history_, self.alpha_updates_, self.stopping_reason_ = [], 0, "max_iter"
        while passes < self.max_passes and iteration < self.max_iter:
            changed = 0
            for second in range(len(self.X)):
                e2 = self._error(second); r2 = e2 * self.y[second]
                if not ((r2 < -self.tolerance and self.alphas[second] < self.C - self.epsilon) or (r2 > self.tolerance and self.alphas[second] > self.epsilon)):
                    continue
                candidates = [i for i in range(len(self.X)) if i != second]
                first = max(candidates, key=lambda i: abs(self._error(i) - e2))
                updated = self._take_step(first, second)
                if not updated:
                    rng.shuffle(candidates)
                    for first in candidates:
                        if self._take_step(first, second):
                            updated = True
                            break
                changed += int(updated)
            # Count whether any KKT-violating pair changed by comparing before/after is costly;
            # derive it from a constraint scan after a full pass instead.
            violations = self.kkt_violations()
            passes = passes + 1 if violations == 0 else 0
            iteration += 1
            self.alpha_updates_ += changed
            self.history_.append({"iteration": iteration, "dual_objective": self.dual_objective(),
                                  "alpha_updates": changed, "cumulative_alpha_updates": self.alpha_updates_,
                                  "kkt_violations": violations, "max_kkt_residual": self.max_kkt_residual(),
                                  "support_vectors": len(self.support_indices()),
                                  "equality_residual": self.equality_residual(),
                                  "max_box_constraint_violation": self.max_box_constraint_violation()})
        self.iterations_, self.passes_without_change_, self.converged_ = iteration, passes, passes >= self.max_passes
        if self.converged_:
            self.stopping_reason_ = "kkt_satisfied_for_max_passes"
        return self

    def decision_function(self, X):
        kernel = resolve(self.kernel_name, self.gamma, self.degree, self.coef0)
        return [self.bias + sum(alpha * label * kernel(support, row)
                                for alpha, label, support in zip(self.alphas, self.y, self.X) if alpha > self.epsilon)
                for row in X]

    def predict_signed(self, X):
        return [1 if score >= 0.0 else -1 for score in self.decision_function(X)]

    def support_indices(self):
        return [index for index, alpha in enumerate(self.alphas) if alpha > self.epsilon]

    def equality_residual(self):
        return sum(alpha * label for alpha, label in zip(self.alphas, self.y))

    def kkt_violations(self):
        violations = 0
        for index, (alpha, label) in enumerate(zip(self.alphas, self.y)):
            margin = label * self._score_index(index)
            if alpha <= self.epsilon and margin < 1.0 - self.tolerance:
                violations += 1
            elif alpha >= self.C - self.epsilon and margin > 1.0 + self.tolerance:
                violations += 1
            elif self.epsilon < alpha < self.C - self.epsilon and abs(margin - 1.0) > self.tolerance:
                violations += 1
        return violations
