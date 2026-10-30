import math
import unittest

from src.kernels import linear, polynomial, rbf
from src.linalg import add, dot, matvec, norm, scale, squared_distance, subtract
from src.metrics import classification_report, confusion_matrix
from src.svc import SVC
from src.svm_primal import PrimalLinearSVM
from src.text_vectorizer import TfidfVectorizer


class LinearAlgebraTests(unittest.TestCase):
    def test_operations(self):
        self.assertEqual(dot([1, 2], [3, 4]), 11)
        self.assertEqual(add([1, 2], [3, 4]), [4, 6])
        self.assertEqual(subtract([3, 4], [1, 2]), [2, 2])
        self.assertEqual(scale([1, 2], 3), [3, 6])
        self.assertEqual(squared_distance([1, 2], [4, 6]), 25)
        self.assertAlmostEqual(norm([3, 4]), 5.0)
        self.assertEqual(matvec([[1, 2], [3, 4]], [2, 1]), [4, 10])


class KernelTests(unittest.TestCase):
    def test_kernel_symmetry_and_values(self):
        a, b = [1.0, 2.0], [3.0, 4.0]
        self.assertEqual(linear(a, b), 11.0)
        self.assertEqual(polynomial(a, b, gamma=1.0, degree=2, coef0=1.0), 144.0)
        self.assertAlmostEqual(rbf(a, b, gamma=0.5), rbf(b, a, gamma=0.5))
        self.assertAlmostEqual(rbf(a, a, gamma=0.5), 1.0)


class OptimizationTests(unittest.TestCase):
    X = [[-2.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [2.0, 1.0]]
    signed = [-1, -1, 1, 1]

    def test_primal_gradient_finite_difference(self):
        model = PrimalLinearSVM(C=0.4, learning_rate=0.01, epochs=1)
        model.weights, model.bias = [0.2, -0.1], 0.3
        gradient, _ = model.gradient(self.X, self.signed)
        epsilon = 1e-6
        for coordinate in range(2):
            baseline = model.weights[coordinate]
            model.weights[coordinate] = baseline + epsilon; up = model.objective(self.X, self.signed)
            model.weights[coordinate] = baseline - epsilon; down = model.objective(self.X, self.signed)
            model.weights[coordinate] = baseline
            self.assertAlmostEqual(gradient[coordinate], (up - down) / (2 * epsilon), places=4)

    def test_primal_and_dual_linear_fit_and_label_mapping(self):
        for optimizer in ("primal", "smo"):
            model = SVC(C=1.0, kernel="linear", optimizer=optimizer, learning_rate=0.003, max_iter=1500, random_state=4).fit(self.X, [0, 0, 1, 1])
            self.assertEqual(model.predict(self.X), [0, 0, 1, 1])
            self.assertEqual(model.classes_, [0, 1])

    def test_smo_constraints_and_support_vectors(self):
        model = SVC(C=1.0, kernel="linear", optimizer="smo", max_iter=3000, random_state=4).fit(self.X, [0, 0, 1, 1])
        self.assertTrue(model.support_indices_)
        self.assertLessEqual(abs(model.diagnostics_["equality_residual"]), 1e-4)
        self.assertLessEqual(model.diagnostics_["kkt_violations"], 1)
        self.assertTrue(model.diagnostics_["history"])
        self.assertGreaterEqual(model.diagnostics_["alpha_updates"], 1)
        self.assertLessEqual(model.diagnostics_["max_box_constraint_violation"], 1e-10)

    def test_xor_requires_nonlinear_kernel(self):
        X, y = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]], [0, 1, 1, 0]
        linear_model = SVC(C=10.0, kernel="linear", optimizer="smo", max_iter=5000, random_state=4).fit(X, y)
        rbf_model = SVC(C=10.0, kernel="rbf", gamma=2.0, optimizer="smo", max_iter=5000, random_state=4).fit(X, y)
        self.assertNotEqual(linear_model.predict(X), y)
        self.assertEqual(rbf_model.predict(X), y)

    def test_duplicates_overlap_and_imbalance_are_handled(self):
        X = [[-2., -1.], [-2., -1.], [-1., -1.], [1., 1.], [2., 1.], [3., 2.]]
        y = [0, 0, 0, 0, 0, 1]
        model = SVC(C=1.0, kernel="linear", optimizer="smo", max_iter=3000, random_state=4).fit(X, y)
        prediction = model.predict([[0., 0.], [3., 2.]])
        self.assertEqual(len(prediction), 2)
        self.assertLessEqual(abs(model.diagnostics_["equality_residual"]), 1e-4)


class TextAndMetricTests(unittest.TestCase):
    def test_train_test_isolation_and_unknown_tokens(self):
        vectorizer = TfidfVectorizer(max_features=10).fit(["alpha beta", "alpha gamma"])
        self.assertNotIn("unseen", vectorizer.vocabulary_)
        transformed = vectorizer.transform(["unseen alpha"])
        self.assertGreater(sum(transformed[0]), 0.0)
        self.assertEqual(vectorizer.transform(["unseen"])[0], [0.0] * len(vectorizer.vocabulary_))
        self.assertIsInstance(vectorizer.transform(["alpha"], sparse=True)[0], dict)

    def test_metrics(self):
        labels, matrix = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], labels=[0, 1])
        self.assertEqual(labels, [0, 1]); self.assertEqual(matrix, [[1, 1], [0, 2]])
        report = classification_report([0, 0, 1, 1], [0, 1, 1, 1], labels=[0, 1])
        self.assertAlmostEqual(report["accuracy"], 0.75)
        self.assertAlmostEqual(report["macro_f1"], (2 / 3 + 0.8) / 2)
        self.assertAlmostEqual(report["weighted_f1"], (2 / 3 + 0.8) / 2)
        self.assertAlmostEqual(report["balanced_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
