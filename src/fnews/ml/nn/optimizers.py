from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class Optimizer:
    def step(self, params_and_grads: list[tuple[np.ndarray, np.ndarray]]) -> None:
        raise NotImplementedError


@dataclass
class SGD(Optimizer):
    learning_rate: float = 0.01

    def step(self, params_and_grads: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for param, grad in params_and_grads:
            param -= self.learning_rate * grad


@dataclass
class SGDMomentum(Optimizer):
    learning_rate: float = 0.01
    momentum: float = 0.9
    velocity: dict[int, np.ndarray] = field(default_factory=dict)

    def step(self, params_and_grads: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for param, grad in params_and_grads:
            key = id(param)
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(param)
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grad
            param += self.velocity[key]


@dataclass
class Adam(Optimizer):
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    t: int = 0
    m: dict[int, np.ndarray] = field(default_factory=dict)
    v: dict[int, np.ndarray] = field(default_factory=dict)

    def step(self, params_and_grads: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.t += 1
        for param, grad in params_and_grads:
            key = id(param)
            if key not in self.m:
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad * grad)
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
