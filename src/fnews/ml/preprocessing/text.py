from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+")
PUNCT_RE = re.compile(r"[^a-zA-Z0-9'\s]+")
SPACE_RE = re.compile(r"\s+")


def normalize_text(
    text: object,
    *,
    lowercase: bool = True,
    normalize_urls: bool = True,
    normalize_numbers: bool = False,
    normalize_punctuation: bool = False,
) -> str:
    value = "" if text is None else str(text)
    if normalize_urls:
        value = URL_RE.sub(" URL ", value)
    if lowercase:
        value = value.lower()
    if normalize_numbers:
        value = re.sub(r"\d+", " NUMBER ", value)
    if normalize_punctuation:
        value = PUNCT_RE.sub(" ", value)
    return SPACE_RE.sub(" ", value).strip()


def tokenize(text: object, *, mode: str = "minimal") -> list[str]:
    if mode == "minimal":
        normalized = normalize_text(text, normalize_punctuation=False)
    elif mode == "punct":
        normalized = normalize_text(text, normalize_punctuation=True)
    else:
        raise ValueError(f"Unknown tokenization mode: {mode}")
    return TOKEN_RE.findall(normalized)


def build_vocabulary(
    documents: list[list[str]],
    *,
    min_frequency: int = 2,
    max_vocabulary_size: int | None = 20_000,
    unknown_token: str = "<UNK>",
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for tokens in documents:
        counts.update(tokens)

    terms = [
        (term, count)
        for term, count in counts.items()
        if count >= min_frequency and term != unknown_token
    ]
    terms.sort(key=lambda item: (-item[1], item[0]))
    if max_vocabulary_size is not None:
        terms = terms[: max(0, max_vocabulary_size - 1)]

    vocabulary = {unknown_token: 0}
    for term, _ in terms:
        vocabulary[term] = len(vocabulary)
    return vocabulary


def tokens_to_indices(tokens: list[str], vocabulary: dict[str, int], unknown_token: str = "<UNK>") -> list[int]:
    unknown_index = vocabulary[unknown_token]
    return [vocabulary.get(token, unknown_index) for token in tokens]


def save_vocabulary(vocabulary: dict[str, int], path: str | Path) -> None:
    Path(path).write_text(json.dumps(vocabulary, indent=2, sort_keys=True), encoding="utf-8")


def load_vocabulary(path: str | Path) -> dict[str, int]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
