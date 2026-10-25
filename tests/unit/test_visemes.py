from __future__ import annotations

import pytest

from train_speaking.voice.visemes import MIN_GAP_FOR_REST_S, REST, phoneme_visemes, viseme_track


@pytest.mark.parametrize(
    ("phonemes", "expected"),
    [
        ("bˈɑ", ["closed", "open"]),
        ("fˈi", ["teeth", "wide"]),
        ("hˈɛlO", ["small", "mid", "small", "round"]),
        ("I", ["open"]),
        ("Q", ["round"]),
        ("ˈˌ ", []),
    ],
)
def test_phoneme_visemes(phonemes: str, expected: list[str]) -> None:
    assert phoneme_visemes(phonemes) == expected


def test_track_spreads_phonemes_and_ends_at_rest() -> None:
    track = viseme_track([(0.0, 0.4, "bɑ")])
    assert track == [(0.0, "closed"), (0.2, "open"), (0.4, REST)]


def test_track_inserts_rest_between_words_with_gap() -> None:
    gap_start = 0.3
    track = viseme_track([(0.0, gap_start, "mɑ"), (gap_start + MIN_GAP_FOR_REST_S + 0.1, 0.8, "i")])
    assert (gap_start, REST) in track
    assert track[-1] == (0.8, REST)


def test_track_collapses_repeated_shapes_and_handles_empty_input() -> None:
    assert viseme_track([(0.0, 0.2, "mb"), (0.2, 0.4, "p")]) == [(0.0, "closed"), (0.4, REST)]
    assert viseme_track([]) == [(0.0, REST)]
    assert viseme_track([(0.5, 0.5, "ɑ")]) == [(0.0, REST)]
