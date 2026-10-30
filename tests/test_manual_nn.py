import numpy as np

from fnews.ml.models.mlp import ManualMLPClassifier
from fnews.ml.nn.activations import relu, sigmoid, softmax, tanh
from fnews.ml.nn.gradient_check import numerical_gradient, relative_error
from fnews.ml.nn.layers import Dense
from fnews.ml.nn.losses import binary_cross_entropy, binary_cross_entropy_backward
from fnews.ml.nn.optimizers import SGD, Adam


def test_activation_values_are_stable() -> None:
    values = np.array([[-1000.0, 0.0, 1000.0]])

    assert np.allclose(relu(values), [[0.0, 0.0, 1000.0]])
    assert np.allclose(sigmoid(values), [[0.0, 0.5, 1.0]])
    assert np.allclose(tanh(np.array([[0.0]])), [[0.0]])
    assert np.allclose(softmax(np.array([[1.0, 2.0]])).sum(axis=1), [1.0])


def test_dense_gradient_check() -> None:
    layer = Dense(2, 1, seed=7)
    inputs = np.array([[0.2, -0.4], [0.7, 0.3]])
    targets = np.array([[1.0], [0.0]])

    def loss_fn() -> float:
        predictions = layer.forward(inputs)
        return float(np.mean((predictions - targets) ** 2))

    predictions = layer.forward(inputs)
    grad_output = 2 * (predictions - targets) / predictions.size
    layer.backward(grad_output)
    numerical = numerical_gradient(loss_fn, layer.weights)

    assert relative_error(layer.grad_weights, numerical) < 1e-7


def test_binary_cross_entropy_backward_shape() -> None:
    probs = np.array([[0.8], [0.2]])
    labels = np.array([[1.0], [0.0]])
    grad = binary_cross_entropy_backward(probs, labels)

    assert np.isclose(binary_cross_entropy(probs, labels), -np.log(0.8))
    assert grad.shape == probs.shape


def test_optimizers_move_parameters() -> None:
    param = np.array([1.0])
    SGD(learning_rate=0.1).step([(param, np.array([0.5]))])
    assert np.allclose(param, [0.95])

    param = np.array([1.0])
    Adam(learning_rate=0.1).step([(param, np.array([0.5]))])
    assert param[0] < 1.0


def test_mlp_learns_tiny_separable_problem() -> None:
    features = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    labels = np.array([0, 0, 1, 1])
    model = ManualMLPClassifier(
        input_dim=2,
        hidden_dims=(4,),
        learning_rate=0.02,
        seed=5,
        optimizer_name="adam",
    )
    model.fit(features, labels, epochs=300, batch_size=4)

    assert model.predict(features).tolist() == labels.tolist()
