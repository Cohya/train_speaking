"""Filler-word detection from transcript text.

Definite fillers (um, uh) are always disfluencies. Words like "like" or "basically" are
only sometimes fillers, so they are reported separately as "possible" and never scored
on their own.
"""

from __future__ import annotations

import re

from train_speaking.models.transcript import FillerCounts

DEFINITE_FILLERS = frozenset({"um", "umm", "uh", "uhh", "uhm", "er", "erm", "hmm", "mm", "ah"})
POSSIBLE_FILLER_PHRASES = (
    "you know",
    "i mean",
    "sort of",
    "kind of",
    "like",
    "basically",
    "actually",
    "literally",
)
_TOKEN = re.compile(r"[a-z']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def count_fillers(text: str) -> FillerCounts:
    tokens = tokenize(text)
    definite: dict[str, int] = {}
    for token in tokens:
        if token in DEFINITE_FILLERS:
            definite[token] = definite.get(token, 0) + 1
    joined = f" {' '.join(tokens)} "
    possible: dict[str, int] = {}
    for phrase in POSSIBLE_FILLER_PHRASES:
        count = joined.count(f" {phrase} ")
        if count:
            possible[phrase] = count
    return FillerCounts(definite=definite, possible=possible, total_words=len(tokens))
