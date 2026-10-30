import numpy as np

from fnews.ml.features.bow import count_vectorize_indexed_documents
from fnews.ml.features.image_features import (
    dense_rows_to_sparse,
    rgb_histogram,
    rgb_to_luminance,
    sobel_edge_magnitude,
)
from fnews.ml.features.tfidf import ManualTfidfVectorizer
from fnews.ml.metrics.classification import (
    binary_classification_metrics,
    confusion_matrix_binary,
    roc_auc_score_manual,
)
from fnews.ml.models.linear_svm import ManualLinearSVM, labels_to_signed
from fnews.ml.models.logistic_regression import ManualLogisticRegression, sigmoid_stable
from fnews.ml.preprocessing.scaler import ManualStandardScaler
from fnews.ml.preprocessing.split import (
    random_split_indices,
    stratified_sample_indices,
    stratified_split_indices,
)
from fnews.ml.preprocessing.text import build_vocabulary, tokenize


def test_tokenizer_and_vocabulary() -> None:
    docs = [tokenize("Hello, HELLO world!"), tokenize("World 123")]
    vocab = build_vocabulary(docs, min_frequency=1, max_vocabulary_size=4)

    assert docs == [["hello", "hello", "world"], ["world", "123"]]
    assert vocab["<UNK>"] == 0
    assert "hello" in vocab
    assert "world" in vocab


def test_bow_counts_and_binary_mode() -> None:
    docs = [[1, 2, 1], [2, 3]]

    assert count_vectorize_indexed_documents(docs) == [{1: 2.0, 2: 1.0}, {2: 1.0, 3: 1.0}]
    assert count_vectorize_indexed_documents(docs, binary=True) == [{1: 1.0, 2: 1.0}, {2: 1.0, 3: 1.0}]


def test_tfidf_hand_calculation_without_l2() -> None:
    vectorizer = ManualTfidfVectorizer(min_frequency=1, max_vocabulary_size=10, l2_normalize=False)
    vectors = vectorizer.fit_transform(["cat cat dog", "dog bird"])

    assert vectorizer.vocabulary is not None
    assert vectorizer.idf is not None
    cat = vectorizer.vocabulary["cat"]
    dog = vectorizer.vocabulary["dog"]
    expected_cat_idf = np.log((2 + 1) / (1 + 1)) + 1
    expected_dog_idf = np.log((2 + 1) / (2 + 1)) + 1
    assert np.isclose(vectorizer.idf[cat], expected_cat_idf)
    assert np.isclose(vectors[0][cat], 2 * expected_cat_idf)
    assert np.isclose(vectors[0][dog], expected_dog_idf)


def test_splitting_reproducible_no_overlap_and_stratified() -> None:
    labels = np.array([0] * 80 + [1] * 20)
    a, b = random_split_indices(100, (0.7, 0.3), seed=7)
    assert len(set(a) & set(b)) == 0
    assert len(a) == 70
    assert np.array_equal(a, random_split_indices(100, (0.7, 0.3), seed=7)[0])

    train, val = stratified_split_indices(labels, (0.75, 0.25), seed=7)
    assert len(set(train) & set(val)) == 0
    assert abs(labels[train].mean() - labels.mean()) < 0.03

    sample = stratified_sample_indices(labels, 10, seed=7)
    assert len(sample) == 10
    assert labels[sample].sum() == 2


def test_metrics_hand_computed() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_score = np.array([0.1, 0.8, 0.7, 0.9])
    metrics = binary_classification_metrics(y_true, y_pred, y_score)

    assert np.array_equal(confusion_matrix_binary(y_true, y_pred), np.array([[1, 1], [0, 2]]))
    assert np.isclose(metrics["accuracy"], 0.75)
    assert np.isclose(metrics["precision"], 2 / 3)
    assert np.isclose(metrics["recall"], 1.0)
    assert np.isclose(roc_auc_score_manual(y_true, y_score), 0.75)


def test_logistic_regression_learns_tiny_problem() -> None:
    vectors = [{0: 1.0}, {0: 2.0}, {1: 1.0}, {1: 2.0}]
    labels = np.array([0, 0, 1, 1])
    model = ManualLogisticRegression(n_features=2, learning_rate=0.5, seed=3)
    model.fit(vectors, labels, epochs=40, batch_size=2, early_stopping_patience=None)

    predictions = model.predict(vectors)
    assert predictions.tolist() == labels.tolist()


def test_linear_svm_learns_tiny_problem() -> None:
    vectors = [{0: 1.0}, {0: 2.0}, {1: 1.0}, {1: 2.0}]
    labels = np.array([0, 0, 1, 1])
    model = ManualLinearSVM(n_features=2, learning_rate=0.5, l2=0.0, seed=3)
    model.fit(vectors, labels, epochs=40, batch_size=2, early_stopping_patience=None)

    predictions = model.predict(vectors)
    assert predictions.tolist() == labels.tolist()
    assert labels_to_signed(labels).tolist() == [-1.0, -1.0, 1.0, 1.0]


def test_linear_svm_hinge_loss_hand_computed() -> None:
    vectors = [{0: 1.0}, {0: -1.0}]
    labels = np.array([1, 0])
    model = ManualLinearSVM(n_features=1, learning_rate=0.1, l2=0.0)
    model.weights[0] = 0.5

    assert np.isclose(model.hinge_loss(vectors, labels), 0.5)


def test_sigmoid_stable_extreme_values() -> None:
    values = sigmoid_stable(np.array([-1000.0, 0.0, 1000.0]))

    assert np.isclose(values[0], 0.0)
    assert np.isclose(values[1], 0.5)
    assert np.isclose(values[2], 1.0)


def test_manual_standard_scaler() -> None:
    values = np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]])
    scaled = ManualStandardScaler().fit_transform(values)

    assert np.allclose(scaled[:, 0].mean(), 0.0)
    assert np.allclose(scaled[:, 0].std(), 1.0)
    assert np.allclose(scaled[:, 1], 0.0)


def test_handcrafted_image_helpers() -> None:
    image = np.zeros((2, 2, 3), dtype=float)
    image[:, :, 0] = np.array([[0.0, 1.0], [0.5, 0.5]])
    histogram = rgb_histogram(image, bins=2)
    luminance = rgb_to_luminance(image)
    edges = sobel_edge_magnitude(luminance)
    sparse = dense_rows_to_sparse(np.array([[0.0, 2.0, -1.0]]))

    assert np.allclose(histogram[:2], [0.25, 0.75])
    assert luminance.shape == (2, 2)
    assert edges.shape == (2, 2)
    assert sparse == [{1: 2.0, 2: -1.0}]
