from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fnews.ml.features.bow import SparseVector, count_vectorize_indexed_documents
from fnews.ml.preprocessing.text import build_vocabulary, tokenize, tokens_to_indices


@dataclass
class ManualTfidfVectorizer:
    min_frequency: int = 2
    max_vocabulary_size: int | None = 20_000
    tokenization_mode: str = "minimal"
    l2_normalize: bool = True
    unknown_token: str = "<UNK>"

    vocabulary: dict[str, int] | None = None
    document_frequencies: np.ndarray | None = None
    idf: np.ndarray | None = None

    def fit(self, texts: list[object]) -> ManualTfidfVectorizer:
        tokenized = [tokenize(text, mode=self.tokenization_mode) for text in texts]
        self.vocabulary = build_vocabulary(
            tokenized,
            min_frequency=self.min_frequency,
            max_vocabulary_size=self.max_vocabulary_size,
            unknown_token=self.unknown_token,
        )
        indexed = [tokens_to_indices(tokens, self.vocabulary, self.unknown_token) for tokens in tokenized]
        n_features = len(self.vocabulary)
        df = np.zeros(n_features, dtype=np.float64)
        for doc in indexed:
            for index in set(doc):
                df[index] += 1
        n_docs = len(indexed)
        self.document_frequencies = df
        self.idf = np.log((n_docs + 1) / (df + 1)) + 1
        return self

    def transform(self, texts: list[object]) -> list[SparseVector]:
        self._require_fit()
        assert self.vocabulary is not None
        assert self.idf is not None
        tokenized = [tokenize(text, mode=self.tokenization_mode) for text in texts]
        indexed = [tokens_to_indices(tokens, self.vocabulary, self.unknown_token) for tokens in tokenized]
        counts = count_vectorize_indexed_documents(indexed)
        vectors: list[SparseVector] = []
        for vector in counts:
            tfidf = {index: value * float(self.idf[index]) for index, value in vector.items()}
            if self.l2_normalize:
                norm = math.sqrt(sum(value * value for value in tfidf.values()))
                if norm > 0:
                    tfidf = {index: value / norm for index, value in tfidf.items()}
            vectors.append(tfidf)
        return vectors

    def fit_transform(self, texts: list[object]) -> list[SparseVector]:
        self.fit(texts)
        return self.transform(texts)

    @property
    def n_features(self) -> int:
        self._require_fit()
        assert self.vocabulary is not None
        return len(self.vocabulary)

    def save(self, directory: str | Path) -> None:
        self._require_fit()
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        assert self.vocabulary is not None
        assert self.document_frequencies is not None
        assert self.idf is not None
        metadata = {
            "min_frequency": self.min_frequency,
            "max_vocabulary_size": self.max_vocabulary_size,
            "tokenization_mode": self.tokenization_mode,
            "l2_normalize": self.l2_normalize,
            "unknown_token": self.unknown_token,
            "vocabulary": self.vocabulary,
        }
        (path / "tfidf_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        np.savez(
            path / "tfidf_arrays.npz",
            document_frequencies=self.document_frequencies,
            idf=self.idf,
        )

    @classmethod
    def load(cls, directory: str | Path) -> ManualTfidfVectorizer:
        path = Path(directory)
        metadata = json.loads((path / "tfidf_metadata.json").read_text(encoding="utf-8"))
        arrays = np.load(path / "tfidf_arrays.npz")
        vectorizer = cls(
            min_frequency=metadata["min_frequency"],
            max_vocabulary_size=metadata["max_vocabulary_size"],
            tokenization_mode=metadata["tokenization_mode"],
            l2_normalize=metadata["l2_normalize"],
            unknown_token=metadata["unknown_token"],
        )
        vectorizer.vocabulary = {key: int(value) for key, value in metadata["vocabulary"].items()}
        vectorizer.document_frequencies = arrays["document_frequencies"]
        vectorizer.idf = arrays["idf"]
        return vectorizer

    def _require_fit(self) -> None:
        if self.vocabulary is None or self.idf is None:
            raise RuntimeError("Vectorizer is not fitted")
