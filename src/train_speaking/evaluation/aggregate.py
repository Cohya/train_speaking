"""Session-level aggregation of per-answer measurements."""

from __future__ import annotations

from train_speaking.evaluation.text_metrics import content_tokens, mtld
from train_speaking.models.report import SessionMetricsSummary
from train_speaking.models.session import Session


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def summarize_metrics(session: Session) -> SessionMetricsSummary:
    answers = session.candidate_turns()
    spoken = [a for a in answers if not a.is_text_answer and a.metrics is not None]
    total_words = sum(a.metrics.word_count for a in answers if a.metrics is not None)
    spoken_words = sum(a.metrics.word_count for a in spoken if a.metrics is not None)
    definite = sum(a.metrics.fillers.definite_total for a in spoken if a.metrics is not None)
    possible = sum(a.metrics.fillers.possible_total for a in spoken if a.metrics is not None)
    speaking_s = sum(a.metrics.duration_s for a in spoken if a.metrics is not None)
    long_pauses = sum(a.metrics.pause_count_gt_2s for a in spoken if a.metrics is not None)
    pauses_1s = sum(a.metrics.pause_count_gt_1s for a in spoken if a.metrics is not None)

    wpm_weighted = [
        (a.metrics.wpm, a.metrics.word_count)
        for a in spoken
        if a.metrics is not None and a.metrics.wpm is not None
    ]
    weight = sum(w for _, w in wpm_weighted)
    mean_wpm = round(sum(v * w for v, w in wpm_weighted) / weight, 1) if weight else None

    return SessionMetricsSummary(
        answer_count=len(answers),
        total_speaking_s=round(speaking_s, 1),
        total_words=total_words,
        mean_wpm=mean_wpm,
        fillers_per_100_words=round(100 * definite / spoken_words, 2) if spoken_words else None,
        possible_fillers_per_100_words=(
            round(100 * possible / spoken_words, 2) if spoken_words else None
        ),
        pauses_gt_1s_per_min=round(pauses_1s / (speaking_s / 60), 2) if speaking_s else None,
        pauses_gt_2s_per_min=round(long_pauses / (speaking_s / 60), 2) if speaking_s else None,
        longest_pause_s=max((a.metrics.longest_pause_s for a in spoken if a.metrics), default=0.0),
        mean_response_latency_s=_mean(
            [
                a.metrics.response_latency_s
                for a in spoken
                if a.metrics is not None and a.metrics.response_latency_s is not None
            ]
        ),
        lexical_diversity_mtld=mtld([t for a in answers for t in content_tokens(a.text)]),
        mean_word_probability=_mean(
            [
                a.metrics.mean_word_probability
                for a in spoken
                if a.metrics is not None and a.metrics.mean_word_probability is not None
            ]
        ),
    )
