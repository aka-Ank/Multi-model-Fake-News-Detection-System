from __future__ import annotations

import numpy as np


def seeded_shuffle_indices(n_items: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_items)
    rng.shuffle(indices)
    return indices


def random_split_indices(n_items: int, fractions: tuple[float, ...], seed: int) -> tuple[np.ndarray, ...]:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to 1.0")
    indices = seeded_shuffle_indices(n_items, seed)
    sizes = [int(n_items * fraction) for fraction in fractions[:-1]]
    sizes.append(n_items - sum(sizes))
    output = []
    start = 0
    for size in sizes:
        output.append(indices[start : start + size])
        start += size
    return tuple(output)


def stratified_split_indices(
    labels: np.ndarray,
    fractions: tuple[float, ...],
    seed: int,
) -> tuple[np.ndarray, ...]:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to 1.0")

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    buckets = [[] for _ in fractions]
    for label in sorted(set(labels.tolist())):
        label_indices = np.flatnonzero(labels == label)
        rng.shuffle(label_indices)
        sizes = [int(len(label_indices) * fraction) for fraction in fractions[:-1]]
        sizes.append(len(label_indices) - sum(sizes))
        start = 0
        for bucket, size in zip(buckets, sizes, strict=True):
            bucket.extend(label_indices[start : start + size].tolist())
            start += size

    outputs = []
    for bucket in buckets:
        arr = np.array(bucket, dtype=np.int64)
        rng.shuffle(arr)
        outputs.append(arr)
    return tuple(outputs)


def stratified_sample_indices(labels: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels)
    if sample_size > len(labels):
        raise ValueError("sample_size cannot exceed number of labels")

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    remainders: list[tuple[float, int, np.ndarray]] = []
    for label in sorted(set(labels.tolist())):
        label_indices = np.flatnonzero(labels == label)
        exact = sample_size * len(label_indices) / len(labels)
        take = int(np.floor(exact))
        remainders.append((exact - take, label, label_indices))
        rng.shuffle(label_indices)
        selected.extend(label_indices[:take].tolist())

    remaining = sample_size - len(selected)
    for _, _, label_indices in sorted(remainders, reverse=True)[:remaining]:
        already_selected = set(selected)
        for index in label_indices:
            if int(index) not in already_selected:
                selected.append(int(index))
                break

    output = np.array(selected, dtype=np.int64)
    rng.shuffle(output)
    return output
