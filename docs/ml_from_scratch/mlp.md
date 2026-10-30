# Manual MLP

The first neural model is a small multilayer perceptron implemented with explicit forward and backward passes.

For a dense layer:

```text
Y = XW + b
```

Backward gradients:

```text
dW = X^T dY
db = sum(dY)
dX = dY W^T
```

Supported activations include ReLU, sigmoid, tanh, and stable softmax. Binary classification uses sigmoid output and binary cross entropy.

The Adam optimizer is implemented manually:

```text
m_t = beta1*m_(t-1) + (1-beta1)*g_t
v_t = beta2*v_(t-1) + (1-beta2)*g_t^2
m_hat = m_t / (1-beta1^t)
v_hat = v_t / (1-beta2^t)
theta = theta - lr*m_hat/(sqrt(v_hat)+epsilon)
```

Gradient checking uses central differences:

```text
df/dtheta ≈ (f(theta + epsilon) - f(theta - epsilon)) / (2 epsilon)
```
