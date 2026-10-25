"""Sentence splitting for incremental text-to-speech."""

from __future__ import annotations

import re

_ABBREVIATIONS = frozenset(
    {"e.g.", "i.e.", "etc.", "vs.", "mr.", "mrs.", "ms.", "dr.", "prof.", "inc.", "approx."}
)
_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]?\s+")
MIN_SENTENCE_CHARS = 12


def split_sentences(text: str, min_chars: int = MIN_SENTENCE_CHARS) -> list[str]:
    """Split on sentence-final punctuation, skipping common abbreviations and decimals.

    Fragments shorter than `min_chars` are merged into the following sentence so TTS does
    not produce choppy one-word utterances.
    """
    text = " ".join(text.split())
    if not text:
        return []
    pieces: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        candidate = text[start : match.start() + len(match.group(0).rstrip())].strip()
        last_word = candidate.rsplit(" ", 1)[-1].lower()
        if last_word in _ABBREVIATIONS:
            continue
        pieces.append(candidate)
        start = match.end()
    if start < len(text):
        pieces.append(text[start:].strip())

    merged: list[str] = []
    carry = ""
    for piece in pieces:
        combined = f"{carry} {piece}".strip()
        if len(combined) < min_chars:
            carry = combined
            continue
        merged.append(combined)
        carry = ""
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]} {carry}"
        else:
            merged.append(carry)
    return merged


def limit_sentences(text: str, max_sentences: int) -> str:
    sentences = split_sentences(text, min_chars=1)
    return " ".join(sentences[:max_sentences])
