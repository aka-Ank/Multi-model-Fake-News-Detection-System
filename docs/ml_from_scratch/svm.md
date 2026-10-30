# Linear SVM From Scratch

This milestone implements a binary linear support-vector machine for Fakeddit text baselines.

Labels are mapped from the dataset's binary task to signed labels:

- `0` true -> `-1`
- `1` fake -> `+1`

For a sparse TF-IDF vector `x`, the model computes:

```text
f(x) = w.x + b
```

Prediction uses the sign of the margin:

```text
prediction = 1 if f(x) >= 0 else 0
```

The hinge loss for one example is:

```text
max(0, 1 - y*f(x))
```

The regularized objective is:

```text
mean(max(0, 1 - y_i*f(x_i))) + 0.5 * lambda * ||w||^2
```

For samples violating the margin, the subgradient is:

```text
dw = -y*x
db = -y
```

For samples outside the margin, the hinge contribution is zero. L2 regularization adds:

```text
dw += lambda * w
```

Optimization uses deterministic mini-batch gradient descent with seeded shuffling, optional gradient clipping, and early stopping on validation Macro F1.
