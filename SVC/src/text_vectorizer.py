"""A bounded, train-only TF-IDF vectorizer with no third-party dependencies."""
import math
import re
from collections import Counter


TOKEN = re.compile(r"[A-Za-z0-9_]+(?:'[A-Za-z0-9_]+)?")


class TfidfVectorizer:
    """Uses idf(t) = log((1 + N)/(1 + df(t))) + 1 and optional L2 normalization."""

    def __init__(self, max_features=1000, lowercase=True, l2_normalize=True):
        if max_features < 1:
            raise ValueError("max_features must be positive")
        self.max_features, self.lowercase, self.l2_normalize = max_features, lowercase, l2_normalize
        self.vocabulary_, self.idf_, self.document_count_ = {}, [], 0

    def tokenize(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return TOKEN.findall(text.lower() if self.lowercase else text)

    def fit(self, texts):
        documents = [self.tokenize(text) for text in texts]
        if not documents:
            raise ValueError("at least one training document is required")
        df = Counter(token for tokens in documents for token in set(tokens))
        ordered = sorted(df, key=lambda token: (-df[token], token))[:self.max_features]
        self.vocabulary_ = {token: index for index, token in enumerate(ordered)}
        self.document_count_ = len(documents)
        self.idf_ = [math.log((1.0 + len(documents)) / (1.0 + df[token])) + 1.0 for token in ordered]
        return self

    def transform(self, texts, sparse=False):
        if not self.vocabulary_:
            raise RuntimeError("fit must be called before transform")
        rows = []
        for text in texts:
            counts = Counter(token for token in self.tokenize(text) if token in self.vocabulary_)
            total = sum(counts.values())
            values = {self.vocabulary_[token]: (count / total) * self.idf_[self.vocabulary_[token]]
                      for token, count in counts.items()} if total else {}
            if self.l2_normalize and values:
                magnitude = math.sqrt(sum(value * value for value in values.values()))
                values = {index: value / magnitude for index, value in values.items()}
            rows.append(values if sparse else [values.get(index, 0.0) for index in range(len(self.vocabulary_))])
        return rows


def memory_estimate(rows, columns, bytes_per_float=8):
    return rows * columns * bytes_per_float
