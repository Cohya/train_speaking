"""Turns measurements and judgments into category scores and an overall score."""

from __future__ import annotations

from train_speaking.evaluation.llm_judge import JudgedCategory, JudgeOutput
from train_speaking.evaluation.pronunciation import PronunciationResult
from train_speaking.evaluation.rubric import (
    JUDGED_CATEGORIES,
    MIN_SPEAKING_S,
    MIN_WORDS,
    RUBRIC,
    combine,
    score_fillers,
    score_fluency_pauses,
    score_long_pauses,
    score_mtld,
    score_wpm,
)
from train_speaking.models.config import CATEGORY_LABELS, InterviewConfig
from train_speaking.models.config import EvaluationCategory as C
from train_speaking.models.report import CategoryScore, Evidence, ScoreBasis, SessionMetricsSummary

NOT_ENOUGH_SPEECH = "Not enough spoken audio was captured to measure this reliably."
NEEDS_LLM = "Requires an AI evaluator. Configure an LLM to assess this category."


def _enough_speech(metrics: SessionMetricsSummary) -> bool:
    return metrics.total_speaking_s >= MIN_SPEAKING_S and metrics.total_words >= MIN_WORDS


def _basis(measured: int | None, judged: int | None) -> ScoreBasis:
    if measured is not None and judged is not None:
        return ScoreBasis.HYBRID
    if measured is not None:
        return ScoreBasis.MEASURED
    if judged is not None:
        return ScoreBasis.MODEL_JUDGMENT
    return ScoreBasis.NOT_ASSESSED


def _evidence(judged: JudgedCategory | None) -> list[Evidence]:
    if judged is None:
        return []
    return [Evidence(turn_id=e.turn_id, quote=e.quote) for e in judged.evidence]


def _measured_score(
    category: C, metrics: SessionMetricsSummary
) -> tuple[int | None, str, dict[str, float | int | str]]:
    """Returns (score, feedback, measurements) for the measured part of a category."""
    enough = _enough_speech(metrics)
    if category is C.FILLER_WORDS:
        if not enough or metrics.fillers_per_100_words is None:
            return None, NOT_ENOUGH_SPEECH, {}
        rate = metrics.fillers_per_100_words
        measurements: dict[str, float | int | str] = {"um_uh_per_100_words": rate}
        if metrics.possible_fillers_per_100_words is not None:
            measurements["possible_fillers_per_100_words"] = metrics.possible_fillers_per_100_words
        feedback = (
            f"{rate:.1f} um/uh per 100 words (a lower bound: the recognizer can drop fillers). "
            "Words like 'like' or 'you know' are listed separately and not scored on their own."
        )
        return score_fillers(rate), feedback, measurements
    if category is C.PACE:
        if not enough or metrics.mean_wpm is None or metrics.pauses_gt_2s_per_min is None:
            return None, NOT_ENOUGH_SPEECH, {}
        score = round(
            (score_wpm(metrics.mean_wpm) + score_long_pauses(metrics.pauses_gt_2s_per_min)) / 2
        )
        feedback = (
            f"Average pace {metrics.mean_wpm:.0f} words per minute (120-160 is comfortable for "
            f"interviews); {metrics.pauses_gt_2s_per_min:.1f} pauses over 2 s per minute."
        )
        return (
            score,
            feedback,
            {
                "words_per_minute": metrics.mean_wpm,
                "pauses_over_2s_per_min": metrics.pauses_gt_2s_per_min,
                "longest_pause_s": metrics.longest_pause_s,
            },
        )
    if category is C.FLUENCY:
        if not enough or metrics.pauses_gt_1s_per_min is None:
            return None, "", {}
        rate = metrics.pauses_gt_1s_per_min
        return (
            score_fluency_pauses(rate),
            f"{rate:.1f} pauses over 1 s per minute.",
            {"pauses_over_1s_per_min": rate},
        )
    if category is C.VOCABULARY:
        if metrics.lexical_diversity_mtld is None:
            return None, "", {}
        value = metrics.lexical_diversity_mtld
        return score_mtld(value), f"Lexical diversity (MTLD) {value:.0f}.", {"mtld": value}
    return None, "", {}


def build_category_scores(
    config: InterviewConfig,
    metrics: SessionMetricsSummary,
    judge: JudgeOutput | None,
    pronunciation: PronunciationResult | None = None,
) -> list[CategoryScore]:
    judged_by_name: dict[str, JudgedCategory] = (
        {c.category: c for c in judge.categories} if judge else {}
    )
    scores: list[CategoryScore] = []
    for category in config.criteria:
        label = CATEGORY_LABELS[category]
        if category is C.PRONUNCIATION:
            scores.append(_pronunciation_score(label, metrics, pronunciation))
            continue
        measured, measured_note, measurements = _measured_score(category, metrics)
        judged = judged_by_name.get(category.value) if category in JUDGED_CATEGORIES else None
        judged_score = judged.score if judged else None
        score = combine(measured, judged_score)
        feedback_parts = [p for p in (judged.feedback if judged else "", measured_note) if p]
        if score is None and category in JUDGED_CATEGORIES and not measurements:
            feedback_parts = [NEEDS_LLM]
        scores.append(
            CategoryScore(
                category=category,
                label=label,
                score=score,
                basis=_basis(measured, judged_score),
                evidence=_evidence(judged),
                feedback=" ".join(feedback_parts),
                measurements=measurements,
            )
        )
    return scores


def _pronunciation_score(
    label: str, metrics: SessionMetricsSummary, result: PronunciationResult | None
) -> CategoryScore:
    if result is not None:
        return CategoryScore(
            category=C.PRONUNCIATION,
            label=label,
            score=result.score,
            basis=ScoreBasis.MEASURED,
            evidence=result.evidence,
            feedback=result.feedback,
            measurements=result.measurements,
        )
    measurements: dict[str, float | int | str] = {}
    if metrics.mean_word_probability is not None:
        measurements["asr_word_confidence_informational"] = metrics.mean_word_probability
    return CategoryScore(
        category=C.PRONUNCIATION,
        label=label,
        score=None,
        basis=ScoreBasis.NOT_ASSESSED,
        feedback=RUBRIC[C.PRONUNCIATION].description
        + " Recognizer confidence is shown for information only.",
        measurements=measurements,
    )


def overall_score(config: InterviewConfig, categories: list[CategoryScore]) -> float | None:
    weighted = [(c.score, config.weight_for(c.category)) for c in categories if c.score is not None]
    total_weight = sum(w for _, w in weighted)
    if not total_weight:
        return None
    return round(sum(s * w for s, w in weighted) / total_weight, 1)
