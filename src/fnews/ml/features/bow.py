from __future__ import annotations

from collections import Counter
from typing import TypeAlias

SparseVector: TypeAlias = dict[int, float]


def count_vectorize_indexed_documents(
    documents: list[list[int]],
    *,
    binary: bool = False,
) -> list[SparseVector]:
    """Convert token-index documents to sparse count or presence vectors."""
    vectors: list[SparseVector] = []
    for document in documents:
        counts = Counter(document)
        if binary:
            vectors.append({int(index): 1.0 for index in counts})
        else:
            vectors.append({int(index): float(count) for index, count in counts.items()})
    return vectors
