# Milestone 1: Manual Text Baseline

This milestone implements a text-only baseline without using scikit-learn, PyTorch, TensorFlow, Hugging Face, pretrained embeddings, or ready-made ML components.

## Text Preprocessing

Input field: `clean_title`.

The tokenizer uses Python regular expressions after minimal normalization:

- optional URL normalization
- lowercase
- whitespace normalization
- regex token extraction

## TF-IDF

For each term `t` and document `d`:

```text
tfidf(t, d) = tf(t, d) * idf(t)
idf(t) = log((N + 1) / (df(t) + 1)) + 1
```

Optional L2 normalization:

```text
x = x / sqrt(sum_i x_i^2)
```

The implementation stores sparse vectors as dictionaries:

```text
feature_index -> value
```

## Logistic Regression

Binary model:

```text
z = w.x + b
p = sigmoid(z)
sigmoid(z) = 1 / (1 + exp(-z))
```

Binary cross entropy:

```text
L = -mean(y log(p) + (1-y) log(1-p))
```

The implementation uses a numerically stable BCE form:

```text
max(z, 0) - z*y + log(1 + exp(-abs(z)))
```

Gradients:

```text
dw = X.T @ (p-y) / N + lambda*w
db = mean(p-y)
```

Training uses seeded mini-batch gradient descent with optional gradient clipping, L2 regularization, and early stopping on validation Macro F1.

## Metrics

All metrics are implemented manually:

- confusion matrix
- accuracy
- precision
- recall
- specificity
- F1
- Macro F1
- Weighted F1
- ROC-AUC via trapezoidal integration over a manually generated ROC curve
- Brier score
