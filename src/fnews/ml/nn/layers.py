from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fnews.ml.nn.activations import (
    relu,
    relu_backward,
    sigmoid,
    sigmoid_backward,
    tanh,
    tanh_backward,
)
from fnews.ml.nn.initialization import he_uniform, xavier_uniform


class Layer:
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> list[np.ndarray]:
        return []

    def gradients(self) -> list[np.ndarray]:
        return []


@dataclass
class Dense(Layer):
    input_dim: int
    output_dim: int
    seed: int = 42
    initialization: str = "xavier"
    weights: np.ndarray = field(init=False)
    bias: np.ndarray = field(init=False)
    grad_weights: np.ndarray = field(init=False)
    grad_bias: np.ndarray = field(init=False)
    inputs: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        if self.initialization == "he":
            self.weights = he_uniform(self.input_dim, self.output_dim, rng)
        else:
            self.weights = xavier_uniform(self.input_dim, self.output_dim, rng)
        self.bias = np.zeros((1, self.output_dim), dtype=np.float64)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        del training
        self.inputs = np.asarray(inputs, dtype=np.float64)
        return self.inputs @ self.weights + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.inputs is None:
            raise RuntimeError("Dense.backward called before forward")
        self.grad_weights = self.inputs.T @ grad_output
        self.grad_bias = grad_output.sum(axis=0, keepdims=True)
        return grad_output @ self.weights.T

    def parameters(self) -> list[np.ndarray]:
        return [self.weights, self.bias]

    def gradients(self) -> list[np.ndarray]:
        return [self.grad_weights, self.grad_bias]


class ReLU(Layer):
    def __init__(self) -> None:
        self.inputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        del training
        self.inputs = inputs
        return relu(inputs)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.inputs is None:
            raise RuntimeError("ReLU.backward called before forward")
        return relu_backward(self.inputs, grad_output)


class Sigmoid(Layer):
    def __init__(self) -> None:
        self.outputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        del training
        self.outputs = sigmoid(inputs)
        return self.outputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.outputs is None:
            raise RuntimeError("Sigmoid.backward called before forward")
        return sigmoid_backward(self.outputs, grad_output)


class Tanh(Layer):
    def __init__(self) -> None:
        self.outputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        del training
        self.outputs = tanh(inputs)
        return self.outputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.outputs is None:
            raise RuntimeError("Tanh.backward called before forward")
        return tanh_backward(self.outputs, grad_output)


@dataclass
class Dropout(Layer):
    rate: float
    seed: int = 42
    mask: np.ndarray | None = field(default=None, init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.rate <= 0:
            return inputs
        keep_probability = 1 - self.rate
        self.mask = (self.rng.random(inputs.shape) < keep_probability) / keep_probability
        return inputs * self.mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return grad_output
        return grad_output * self.mask


class Flatten(Layer):
    def __init__(self) -> None:
        self.original_shape: tuple[int, ...] | None = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        del training
        self.original_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.original_shape is None:
            raise RuntimeError("Flatten.backward called before forward")
        return grad_output.reshape(self.original_shape)


@dataclass
class Embedding(Layer):
    vocab_size: int
    embedding_dim: int
    pad_index: int = 0
    seed: int = 42
    weights: np.ndarray = field(init=False)
    grad_weights: np.ndarray = field(init=False)
    token_ids: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        limit = np.sqrt(6.0 / (self.vocab_size + self.embedding_dim))
        self.weights = rng.uniform(-limit, limit, size=(self.vocab_size, self.embedding_dim)).astype(np.float64)
        self.weights[self.pad_index] = 0.0
        self.grad_weights = np.zeros_like(self.weights)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        del training
        self.token_ids = np.asarray(inputs, dtype=np.int64)
        return self.weights[self.token_ids]

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.token_ids is None:
            raise RuntimeError("Embedding.backward called before forward")
        self.grad_weights = np.zeros_like(self.weights)
        np.add.at(self.grad_weights, self.token_ids, grad_output)
        self.grad_weights[self.pad_index] = 0.0
        return np.zeros_like(self.token_ids, dtype=np.float64)

    def parameters(self) -> list[np.ndarray]:
        return [self.weights]

    def gradients(self) -> list[np.ndarray]:
        return [self.grad_weights]


@dataclass
class MaskedMeanPooling(Layer):
    pad_index: int = 0
    token_ids: np.ndarray | None = field(default=None, init=False)
    counts: np.ndarray | None = field(default=None, init=False)
    input_shape: tuple[int, int, int] | None = field(default=None, init=False)

    def forward(self, inputs: np.ndarray, training: bool = True, token_ids: np.ndarray | None = None) -> np.ndarray:
        del training
        if token_ids is None:
            raise ValueError("MaskedMeanPooling requires token_ids")
        self.token_ids = np.asarray(token_ids, dtype=np.int64)
        self.input_shape = inputs.shape
        mask = self.token_ids != self.pad_index
        counts = mask.sum(axis=1, keepdims=True).astype(np.float64)
        counts[counts == 0] = 1.0
        self.counts = counts
        return (inputs * mask[:, :, None]).sum(axis=1) / counts

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.token_ids is None or self.counts is None or self.input_shape is None:
            raise RuntimeError("MaskedMeanPooling.backward called before forward")
        mask = self.token_ids != self.pad_index
        return grad_output[:, None, :] * mask[:, :, None] / self.counts[:, :, None]


@dataclass
class Conv2D(Layer):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    seed: int = 42
    weights: np.ndarray = field(init=False)
    bias: np.ndarray = field(init=False)
    grad_weights: np.ndarray = field(init=False)
    grad_bias: np.ndarray = field(init=False)
    inputs: np.ndarray | None = field(default=None, init=False)
    columns: np.ndarray | None = field(default=None, init=False)
    input_shape: tuple[int, int, int, int] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        fan_out = self.out_channels * self.kernel_size * self.kernel_size
        flat_weights = he_uniform(fan_in, fan_out, rng)[:, : self.out_channels]
        self.weights = flat_weights.T.reshape(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        self.bias = np.zeros((self.out_channels,), dtype=np.float64)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        del training
        self.inputs = np.asarray(inputs, dtype=np.float64)
        self.input_shape = self.inputs.shape
        columns, out_h, out_w = im2col(self.inputs, self.kernel_size, self.stride, self.padding)
        self.columns = columns
        weight_columns = self.weights.reshape(self.out_channels, -1)
        output = columns @ weight_columns.T + self.bias
        n = self.inputs.shape[0]
        return output.reshape(n, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.columns is None or self.input_shape is None:
            raise RuntimeError("Conv2D.backward called before forward")
        grad_rows = grad_output.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        self.grad_bias = grad_rows.sum(axis=0)
        grad_weight_columns = grad_rows.T @ self.columns
        self.grad_weights = grad_weight_columns.reshape(self.weights.shape)
        weight_columns = self.weights.reshape(self.out_channels, -1)
        grad_columns = grad_rows @ weight_columns
        return col2im(grad_columns, self.input_shape, self.kernel_size, self.stride, self.padding)

    def parameters(self) -> list[np.ndarray]:
        return [self.weights, self.bias]

    def gradients(self) -> list[np.ndarray]:
        return [self.grad_weights, self.grad_bias]


@dataclass
class MaxPool2D(Layer):
    pool_size: int = 2
    stride: int = 2
    inputs: np.ndarray | None = field(default=None, init=False)
    max_indices: np.ndarray | None = field(default=None, init=False)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        del training
        self.inputs = np.asarray(inputs, dtype=np.float64)
        n, c, h, w = self.inputs.shape
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        output = np.empty((n, c, out_h, out_w), dtype=np.float64)
        self.max_indices = np.empty((n, c, out_h, out_w), dtype=np.int64)
        for i in range(out_h):
            for j in range(out_w):
                h0 = i * self.stride
                w0 = j * self.stride
                window = self.inputs[:, :, h0 : h0 + self.pool_size, w0 : w0 + self.pool_size]
                flat = window.reshape(n, c, -1)
                self.max_indices[:, :, i, j] = flat.argmax(axis=2)
                output[:, :, i, j] = flat.max(axis=2)
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.inputs is None or self.max_indices is None:
            raise RuntimeError("MaxPool2D.backward called before forward")
        grad_input = np.zeros_like(self.inputs)
        _, _, out_h, out_w = grad_output.shape
        for i in range(out_h):
            for j in range(out_w):
                h0 = i * self.stride
                w0 = j * self.stride
                flat_grad = np.zeros((*grad_output.shape[:2], self.pool_size * self.pool_size), dtype=np.float64)
                np.put_along_axis(
                    flat_grad,
                    self.max_indices[:, :, i, j][..., None],
                    grad_output[:, :, i, j][..., None],
                    axis=2,
                )
                grad_input[:, :, h0 : h0 + self.pool_size, w0 : w0 + self.pool_size] += flat_grad.reshape(
                    *grad_output.shape[:2],
                    self.pool_size,
                    self.pool_size,
                )
        return grad_input


class GlobalAveragePooling2D(Layer):
    def __init__(self) -> None:
        self.input_shape: tuple[int, int, int, int] | None = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        del training
        self.input_shape = inputs.shape
        return inputs.mean(axis=(2, 3))

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.input_shape is None:
            raise RuntimeError("GlobalAveragePooling2D.backward called before forward")
        _, _, h, w = self.input_shape
        return np.repeat(np.repeat(grad_output[:, :, None, None] / (h * w), h, axis=2), w, axis=3)


def im2col(
    inputs: np.ndarray,
    kernel_size: int,
    stride: int,
    padding: int,
) -> tuple[np.ndarray, int, int]:
    n, c, h, w = inputs.shape
    padded = np.pad(inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out_h = (h + 2 * padding - kernel_size) // stride + 1
    out_w = (w + 2 * padding - kernel_size) // stride + 1
    columns = np.empty((n * out_h * out_w, c * kernel_size * kernel_size), dtype=np.float64)
    column_index = 0
    for i in range(out_h):
        for j in range(out_w):
            h0 = i * stride
            w0 = j * stride
            patch = padded[:, :, h0 : h0 + kernel_size, w0 : w0 + kernel_size]
            columns[column_index : column_index + n] = patch.reshape(n, -1)
            column_index += n
    return columns, out_h, out_w


def col2im(
    columns: np.ndarray,
    input_shape: tuple[int, int, int, int],
    kernel_size: int,
    stride: int,
    padding: int,
) -> np.ndarray:
    n, c, h, w = input_shape
    out_h = (h + 2 * padding - kernel_size) // stride + 1
    out_w = (w + 2 * padding - kernel_size) // stride + 1
    padded_grad = np.zeros((n, c, h + 2 * padding, w + 2 * padding), dtype=np.float64)
    row_index = 0
    for i in range(out_h):
        for j in range(out_w):
            h0 = i * stride
            w0 = j * stride
            patch_grad = columns[row_index : row_index + n].reshape(n, c, kernel_size, kernel_size)
            padded_grad[:, :, h0 : h0 + kernel_size, w0 : w0 + kernel_size] += patch_grad
            row_index += n
    if padding == 0:
        return padded_grad
    return padded_grad[:, :, padding:-padding, padding:-padding]
