"""Objective per-answer measurements from word timestamps.

These are measurements, not judgments: words per minute, pauses between words, response
latency, and ASR word confidence. ASR confidence is reported for information only; it
depends on microphone quality, noise, and model bias, and is not a pronunciation score.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

from train_speaking.evaluation.fillers import count_fillers
from train_speaking.models.transcript import AnswerMetrics, WordTiming

PAUSE_THRESHOLD_S = 0.25
LONG_PAUSE_S = 1.0
VERY_LONG_PAUSE_S = 2.0
LOW_CONFIDENCE_PROBABILITY = 0.5
MIN_SPAN_FOR_RATE_S = 2.0


def compute_answer_metrics(
    words: Sequence[WordTiming],
    text: str,
    *,
    speech_started_at: float | None,
    ended_at: float,
    question_ended_at: float | None,
) -> AnswerMetrics:
    """`words` must use session-absolute times (same timeline as the other arguments)."""
    fillers = count_fillers(text)
    word_count = fillers.total_words
    start = words[0].start if words else speech_started_at
    duration = max(0.0, ended_at - start) if start is not None else 0.0

    latency = None
    if question_ended_at is not None and start is not None:
        latency = max(0.0, start - question_ended_at)

    if not words:
        return AnswerMetrics(
            duration_s=round(duration, 2),
            word_count=word_count,
            response_latency_s=_round(latency),
            fillers=fillers,
        )

    gaps = [max(0.0, b.start - a.end) for a, b in pairwise(words)]
    span = words[-1].end - words[0].start
    paused = sum(g for g in gaps if g > PAUSE_THRESHOLD_S)
    wpm = len(words) / (span / 60) if span >= MIN_SPAN_FOR_RATE_S else None
    speaking_time = span - paused
    articulation = (
        len(words) / (speaking_time / 60) if speaking_time >= MIN_SPAN_FOR_RATE_S else None
    )
    probabilities = [w.probability for w in words]
    return AnswerMetrics(
        duration_s=round(duration, 2),
        word_count=word_count,
        wpm=_round(wpm, 1),
        articulation_rate_wpm=_round(articulation, 1),
        pause_count_gt_1s=sum(1 for g in gaps if g > LONG_PAUSE_S),
        pause_count_gt_2s=sum(1 for g in gaps if g > VERY_LONG_PAUSE_S),
        longest_pause_s=round(max(gaps, default=0.0), 2),
        response_latency_s=_round(latency),
        fillers=fillers,
        mean_word_probability=round(sum(probabilities) / len(probabilities), 3),
        low_confidence_word_ratio=round(
            sum(1 for p in probabilities if p < LOW_CONFIDENCE_PROBABILITY) / len(probabilities), 3
        ),
    )


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)
