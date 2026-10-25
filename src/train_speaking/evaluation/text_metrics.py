"""Objective text measurements: lexical diversity and repeated phrases."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from train_speaking.evaluation.fillers import DEFINITE_FILLERS, tokenize

MTLD_THRESHOLD = 0.72
MIN_WORDS_FOR_MTLD = 100
REPEATED_PHRASE_MIN_COUNT = 3
_STOP_NGRAM_WORDS = frozenset({"the", "a", "an", "and", "to", "of", "in", "i", "it", "that"})


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in DEFINITE_FILLERS]


def _mtld_pass(tokens: Sequence[str]) -> float:
    factors = 0.0
    types: set[str] = set()
    count = 0
    for token in tokens:
        types.add(token)
        count += 1
        if len(types) / count <= MTLD_THRESHOLD:
            factors += 1
            types.clear()
            count = 0
    if count:
        ttr = len(types) / count
        factors += (1 - ttr) / (1 - MTLD_THRESHOLD)
    return len(tokens) / factors if factors else float(len(tokens))


def mtld(tokens: Sequence[str]) -> float | None:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis, 2010), bidirectional.

    Returns None for fewer than `MIN_WORDS_FOR_MTLD` tokens, where it is unreliable.
    """
    if len(tokens) < MIN_WORDS_FOR_MTLD:
        return None
    return round((_mtld_pass(tokens) + _mtld_pass(list(reversed(tokens)))) / 2, 1)


def repeated_phrases(texts: Sequence[str], n: int = 3, top: int = 3) -> list[tuple[str, int]]:
    """Most frequent content n-grams used at least `REPEATED_PHRASE_MIN_COUNT` times."""
    counts: Counter[str] = Counter()
    for text in texts:
        tokens = content_tokens(text)
        for i in range(len(tokens) - n + 1):
            gram = tokens[i : i + n]
            if all(word in _STOP_NGRAM_WORDS for word in gram):
                continue
            counts[" ".join(gram)] += 1
    return [(g, c) for g, c in counts.most_common(top) if c >= REPEATED_PHRASE_MIN_COUNT]
