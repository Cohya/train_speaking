from __future__ import annotations

import pytest

from train_speaking.evaluation.audio_metrics import compute_answer_metrics
from train_speaking.evaluation.fillers import count_fillers
from train_speaking.evaluation.text_metrics import MIN_WORDS_FOR_MTLD, mtld, repeated_phrases
from train_speaking.models.transcript import WordTiming


def _words(spec: list[tuple[str, float, float]], probability: float = 0.9) -> list[WordTiming]:
    return [WordTiming(word=w, start=s, end=e, probability=probability) for w, s, e in spec]


def test_count_fillers_separates_definite_and_possible() -> None:
    counts = count_fillers(
        "Um, so I, uh, like, basically built it. You know, it was, umm, like fine."
    )
    assert counts.definite == {"um": 1, "uh": 1, "umm": 1}
    assert counts.possible == {"you know": 1, "like": 2, "basically": 1}
    assert counts.total_words == 15


def test_count_fillers_is_word_boundary_aware() -> None:
    counts = count_fillers("The umbrella was likely useful. I liked it.")
    assert counts.definite == {} and counts.possible == {}


def test_answer_metrics_from_word_timings() -> None:
    # 10 words over 5 s with one 2.5 s pause and one 1.2 s pause.
    spec = [(f"w{i}", float(i * 0.3), float(i * 0.3 + 0.25)) for i in range(6)]
    spec += [("late", 4.25, 4.5), ("more", 4.6, 4.8), ("later", 6.0, 6.3), ("end", 6.35, 6.5)]
    words = _words(spec)
    metrics = compute_answer_metrics(
        words,
        " ".join(w for w, _, _ in spec),
        speech_started_at=0.0,
        ended_at=7.0,
        question_ended_at=-1.5,
    )
    assert metrics.word_count == 10
    assert metrics.pause_count_gt_2s == 1
    assert metrics.pause_count_gt_1s == 2
    assert metrics.longest_pause_s == pytest.approx(2.5)
    assert metrics.wpm == pytest.approx(10 / (6.5 / 60), rel=0.01)
    assert metrics.wpm is not None and metrics.articulation_rate_wpm is not None
    assert metrics.articulation_rate_wpm > metrics.wpm
    assert metrics.response_latency_s == pytest.approx(1.5)
    assert metrics.mean_word_probability == pytest.approx(0.9)


def test_answer_metrics_short_and_empty_answers() -> None:
    short = compute_answer_metrics(
        _words([("yes", 0.0, 0.3)]),
        "yes",
        speech_started_at=0.0,
        ended_at=1.0,
        question_ended_at=None,
    )
    assert short.wpm is None and short.response_latency_s is None
    empty = compute_answer_metrics(
        [], "typed answer here", speech_started_at=None, ended_at=5.0, question_ended_at=None
    )
    assert empty.word_count == 3 and empty.wpm is None and empty.duration_s == 0.0


def test_low_confidence_ratio() -> None:
    words = _words([("a", 0, 0.2), ("b", 0.3, 0.5)], probability=0.3)
    metrics = compute_answer_metrics(
        words, "a b", speech_started_at=0, ended_at=1, question_ended_at=None
    )
    assert metrics.low_confidence_word_ratio == 1.0


def test_mtld_requires_enough_words_and_orders_diversity() -> None:
    assert mtld(["word"] * (MIN_WORDS_FOR_MTLD - 1)) is None
    repetitive = (["we", "did", "the", "thing"] * 50)[:200]
    diverse = [f"word{i}" for i in range(200)]
    low, high = mtld(repetitive), mtld(diverse)
    assert low is not None and high is not None and high > low


def test_repeated_phrases() -> None:
    texts = ["at the end of the day we shipped", "at the end of the day it worked"] * 2
    phrases = dict(repeated_phrases(texts))
    assert phrases.get("end of the") == 4
    assert repeated_phrases(["no repeats in this text at all"]) == []
