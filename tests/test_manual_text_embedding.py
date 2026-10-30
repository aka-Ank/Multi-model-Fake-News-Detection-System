from pathlib import Path

import numpy as np

from fnews.ml.metrics.classification import binary_classification_metrics
from fnews.ml.models.text_embedding import ManualTextEmbeddingClassifier
from fnews.ml.nn.layers import Embedding, MaskedMeanPooling


def test_embedding_forward_backward_repeated_indices_and_pad() -> None:
    layer = Embedding(vocab_size=4, embedding_dim=2, pad_index=0, seed=1)
    token_ids = np.array([[1, 2, 1, 0]])
    output = layer.forward(token_ids)
    layer.backward(np.ones_like(output))

    assert output.shape == (1, 4, 2)
    assert np.allclose(layer.weights[0], 0.0)
    assert np.allclose(layer.grad_weights[0], 0.0)
    assert np.allclose(layer.grad_weights[1], [2.0, 2.0])
    assert np.allclose(layer.grad_weights[2], [1.0, 1.0])


def test_masked_mean_pooling_forward_backward() -> None:
    pooling = MaskedMeanPooling(pad_index=0)
    token_ids = np.array([[1, 2, 0], [0, 0, 0]])
    embeddings = np.array([[[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]], [[9.0, 9.0], [8.0, 8.0], [7.0, 7.0]]])
    pooled = pooling.forward(embeddings, token_ids=token_ids)
    grad = pooling.backward(np.array([[2.0, 4.0], [3.0, 3.0]]))

    assert np.allclose(pooled[0], [3.0, 5.0])
    assert np.allclose(pooled[1], [0.0, 0.0])
    assert np.allclose(grad[0, 0], [1.0, 2.0])
    assert np.allclose(grad[0, 1], [1.0, 2.0])
    assert np.allclose(grad[0, 2], [0.0, 0.0])
    assert np.allclose(grad[1], 0.0)


def test_text_embedding_model_shapes_and_serialization(tmp_path: Path) -> None:
    tokens = np.array([[1, 2, 0], [2, 3, 4]])
    model = ManualTextEmbeddingClassifier(vocab_size=5, embedding_dim=4, hidden_dim=3, seed=2)
    scores = model.predict_proba(tokens)
    model.save(tmp_path)
    loaded = ManualTextEmbeddingClassifier.load(tmp_path)

    assert scores.shape == (2,)
    assert np.allclose(scores, loaded.predict_proba(tokens))


def test_text_embedding_tiny_learning_problem() -> None:
    tokens = np.array([[1, 0, 0], [1, 2, 0], [3, 0, 0], [3, 4, 0]])
    labels = np.array([0, 0, 1, 1])
    model = ManualTextEmbeddingClassifier(
        vocab_size=5,
        embedding_dim=8,
        hidden_dim=8,
        learning_rate=0.01,
        dropout=0.0,
        seed=5,
    )
    history = model.fit(tokens, labels, epochs=100, batch_size=4, shuffle=False, stop_accuracy=1.0)

    assert history[-1]["train_accuracy"] == 1.0


def test_text_embedding_restores_best_validation_epoch() -> None:
    tokens = np.array([[1, 0], [2, 0], [3, 0], [4, 0]])
    labels = np.array([0, 0, 1, 1])
    model = ManualTextEmbeddingClassifier(vocab_size=5, embedding_dim=4, hidden_dim=4, seed=8)
    history = model.fit(
        tokens,
        labels,
        validation_tokens=tokens,
        validation_labels=labels,
        epochs=4,
        batch_size=4,
        shuffle=False,
        early_stopping_patience=1,
        restore_best=True,
    )

    scores = model.predict_proba(tokens)
    restored_f1 = binary_classification_metrics(labels, (scores >= 0.5).astype(int), scores)["macro_f1"]
    assert model.best_epoch_ is not None
    assert np.isclose(restored_f1, max(row["validation_macro_f1"] for row in history))
