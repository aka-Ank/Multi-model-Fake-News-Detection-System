from pathlib import Path

import numpy as np

from fnews.ml.models.cnn import ManualCNNClassifier, ManualFlattenCNNClassifier
from fnews.ml.nn.gradient_check import numerical_gradient, relative_error
from fnews.ml.nn.layers import Conv2D, GlobalAveragePooling2D, MaxPool2D
from fnews.ml.nn.losses import binary_cross_entropy


def test_conv2d_forward_known_example() -> None:
    conv = Conv2D(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)
    conv.weights[:] = np.array([[[[1.0, 0.0], [0.0, -1.0]]]])
    conv.bias[:] = 0.5
    inputs = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])

    output = conv.forward(inputs)

    assert output.shape == (1, 1, 2, 2)
    assert np.allclose(output, [[[[-3.5, -3.5], [-3.5, -3.5]]]])


def test_conv2d_gradient_check_weights_and_input() -> None:
    rng = np.random.default_rng(4)
    conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3, padding=1, seed=5)
    inputs = rng.normal(size=(2, 1, 4, 4))
    grad_output = rng.normal(size=(2, 1, 4, 4))

    def loss_for_weights() -> float:
        return float(np.sum(conv.forward(inputs) * grad_output))

    output = conv.forward(inputs)
    assert output.shape == grad_output.shape
    grad_input = conv.backward(grad_output)
    numerical_w = numerical_gradient(loss_for_weights, conv.weights)

    def loss_for_inputs() -> float:
        return float(np.sum(conv.forward(inputs) * grad_output))

    numerical_x = numerical_gradient(loss_for_inputs, inputs)
    assert relative_error(conv.grad_weights, numerical_w) < 1e-7
    assert relative_error(grad_input, numerical_x) < 1e-7


def test_maxpool_forward_backward() -> None:
    pool = MaxPool2D(pool_size=2, stride=2)
    inputs = np.array([[[[1.0, 3.0], [2.0, 4.0]]]])
    output = pool.forward(inputs)
    grad_input = pool.backward(np.array([[[[10.0]]]]))

    assert np.allclose(output, [[[[4.0]]]])
    assert np.allclose(grad_input, [[[[0.0, 0.0], [0.0, 10.0]]]])


def test_global_average_pooling_forward_backward() -> None:
    gap = GlobalAveragePooling2D()
    inputs = np.array([[[[1.0, 3.0], [5.0, 7.0]]]])
    output = gap.forward(inputs)
    grad_input = gap.backward(np.array([[2.0]]))

    assert np.allclose(output, [[4.0]])
    assert np.allclose(grad_input, np.full_like(inputs, 0.5))


def test_cnn_prediction_shape_and_save_load(tmp_path: Path) -> None:
    model = ManualCNNClassifier(seed=9)
    inputs = np.random.default_rng(0).normal(size=(3, 3, 16, 16))
    scores = model.predict_proba(inputs, batch_size=2)
    model.save(tmp_path)
    loaded = ManualCNNClassifier.load(tmp_path)
    loaded_scores = loaded.predict_proba(inputs, batch_size=2)

    assert scores.shape == (3,)
    assert np.all((scores >= 0) & (scores <= 1))
    assert np.allclose(scores, loaded_scores)


def test_flatten_cnn_learns_tiny_synthetic_images() -> None:
    images = np.zeros((2, 3, 8, 8), dtype=np.float64)
    images[1, :, 2:6, 2:6] = 1.0
    labels = np.array([0, 1])
    model = ManualFlattenCNNClassifier(image_size=8, learning_rate=0.01, seed=2)
    history = model.fit(images, labels, epochs=30, batch_size=2, shuffle=False)

    assert history[-1]["train_accuracy"] == 1.0


def test_bce_probability_consistency() -> None:
    assert binary_cross_entropy(np.array([0.99]), np.array([1])) < 0.02
    assert binary_cross_entropy(np.array([0.01]), np.array([1])) > 4.0
    assert binary_cross_entropy(np.array([0.01]), np.array([0])) < 0.02
    assert binary_cross_entropy(np.array([0.99]), np.array([0])) > 4.0
