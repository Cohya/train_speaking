"""Approximate viseme tracks from TTS phoneme timings, for avatar lip-sync.

Kokoro reports per-token start/end times and IPA phonemes. Each token's phonemes are spread
evenly over its duration and mapped to a small set of mouth shapes that the SVG avatar
draws. This is approximate lip-sync, not phonetically exact animation.
"""

from __future__ import annotations

from collections.abc import Sequence

REST = "rest"
_VISEME_BY_SYMBOL: dict[str, str] = {}
for _symbols, _viseme in (
    ("mbp", "closed"),
    ("fv", "teeth"),
    ("aɑæʌɐAI", "open"),
    ("eɛəɜɚE", "mid"),
    ("iɪjy", "wide"),
    ("oɔuʊwOWQ", "round"),
    ("θðtdnlszʃʒʧʤkgŋhɹrxcq", "small"),
):
    for _symbol in _symbols:
        _VISEME_BY_SYMBOL[_symbol] = _viseme

MIN_GAP_FOR_REST_S = 0.08


def phoneme_visemes(phonemes: str) -> list[str]:
    # Case matters: Kokoro writes diphthongs as capitals (A, I, O, W, Q).
    return [_VISEME_BY_SYMBOL[c] for c in phonemes if c in _VISEME_BY_SYMBOL]


def viseme_track(timeline: Sequence[tuple[float, float, str]]) -> list[tuple[float, str]]:
    """Return (time_s, viseme) change points, starting and ending at rest."""
    track: list[tuple[float, str]] = []
    previous_end = 0.0
    for start, end, phonemes in timeline:
        if start - previous_end >= MIN_GAP_FOR_REST_S:
            track.append((previous_end, REST))
        visemes = phoneme_visemes(phonemes)
        if visemes and end > start:
            step = (end - start) / len(visemes)
            for index, viseme in enumerate(visemes):
                track.append((start + index * step, viseme))
        previous_end = max(previous_end, end)
    track.append((previous_end, REST))

    collapsed: list[tuple[float, str]] = []
    for time_s, viseme in track:
        if collapsed and collapsed[-1][1] == viseme:
            continue
        collapsed.append((round(time_s, 3), viseme))
    return collapsed
