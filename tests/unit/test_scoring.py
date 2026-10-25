from __future__ import annotations

import pytest

from tests.conftest import make_config
from train_speaking.evaluation.llm_judge import JudgedCategory, JudgeOutput, QuotedEvidence
from train_speaking.evaluation.pronunciation import PronunciationResult
from train_speaking.evaluation.rubric import (
    combine,
    score_fillers,
    score_fluency_pauses,
    score_long_pauses,
    score_mtld,
    score_wpm,
)
from train_speaking.evaluation.scoring import (
    NEEDS_LLM,
    NOT_ENOUGH_SPEECH,
    build_category_scores,
    overall_score,
)
from train_speaking.models.config import EvaluationCategory as C
from train_speaking.models.report import ScoreBasis, SessionMetricsSummary

GOOD_METRICS = SessionMetricsSummary(
    answer_count=4,
    total_speaking_s=240,
    total_words=600,
    mean_wpm=140,
    fillers_per_100_words=0.5,
    possible_fillers_per_100_words=2.0,
    pauses_gt_1s_per_min=1.0,
    pauses_gt_2s_per_min=0.0,
    longest_pause_s=1.8,
    lexical_diversity_mtld=75,
    mean_word_probability=0.91,
)


@pytest.mark.parametrize(
    ("wpm", "expected"), [(140, 10), (120, 10), (160, 10), (100, 8), (190, 7), (20, 1)]
)
def test_score_wpm(wpm: float, expected: int) -> None:
    assert score_wpm(wpm) == expected


@pytest.mark.parametrize(
    ("rate", "expected"), [(0, 10), (1.0, 10), (1.5, 9), (4, 7), (6, 5), (14, 2), (30, 1)]
)
def test_score_fillers(rate: float, expected: int) -> None:
    assert score_fillers(rate) == expected


def test_other_bands() -> None:
    assert score_long_pauses(0) == 10 and score_long_pauses(2) == 6 and score_long_pauses(9) == 1
    assert score_fluency_pauses(0) == 10 and score_fluency_pauses(2) == 7
    assert score_mtld(85) == 9 and score_mtld(30) == 4
    assert combine(None, None) is None and combine(8, None) == 8 and combine(None, 6) == 6
    assert combine(9, 6) == 8


def test_without_llm_only_measured_categories_are_scored() -> None:
    scores = {s.category: s for s in build_category_scores(make_config(), GOOD_METRICS, None)}
    assert scores[C.PACE].score == 10 and scores[C.PACE].basis is ScoreBasis.MEASURED
    assert scores[C.FILLER_WORDS].score == 10
    assert scores[C.FLUENCY].basis is ScoreBasis.MEASURED
    assert scores[C.VOCABULARY].score == 8 and scores[C.VOCABULARY].basis is ScoreBasis.MEASURED
    for category in (C.GRAMMAR, C.CLARITY, C.PROFESSIONAL_TONE, C.STRUCTURE):
        assert scores[category].score is None
        assert scores[category].basis is ScoreBasis.NOT_ASSESSED
        assert scores[category].feedback == NEEDS_LLM


def test_pronunciation_never_scored_from_text() -> None:
    scores = {s.category: s for s in build_category_scores(make_config(), GOOD_METRICS, None)}
    pronunciation = scores[C.PRONUNCIATION]
    assert pronunciation.score is None and pronunciation.basis is ScoreBasis.NOT_ASSESSED
    assert "asr_word_confidence_informational" in pronunciation.measurements


def test_pronunciation_result_from_assessor_is_used() -> None:
    result = PronunciationResult(score=7, feedback="Clear overall.", measurements={"pa": 0.8})
    scores = build_category_scores(
        make_config(criteria=("pronunciation",)), GOOD_METRICS, None, result
    )
    assert scores[0].score == 7 and scores[0].basis is ScoreBasis.MEASURED


def test_hybrid_categories_combine_measured_and_judged() -> None:
    judge = JudgeOutput(
        summary="Clear communicator.",
        categories=[
            JudgedCategory(
                category="fluency",
                score=6,
                evidence=[QuotedEvidence(turn_id=2, quote="we we started")],
                feedback="Some restarts.",
            ),
            JudgedCategory(category="grammar", score=7, feedback="Mostly correct."),
        ],
    )
    scores = {s.category: s for s in build_category_scores(make_config(), GOOD_METRICS, judge)}
    fluency = scores[C.FLUENCY]
    assert fluency.basis is ScoreBasis.HYBRID
    assert fluency.score == combine(score_fluency_pauses(1.0), 6)
    assert fluency.evidence[0].quote == "we we started"
    assert "Some restarts." in fluency.feedback and "pauses over 1 s" in fluency.feedback
    assert scores[C.GRAMMAR].basis is ScoreBasis.MODEL_JUDGMENT and scores[C.GRAMMAR].score == 7


def test_insufficient_speech_is_not_scored() -> None:
    metrics = SessionMetricsSummary(
        answer_count=1, total_speaking_s=5, total_words=10, mean_wpm=150
    )
    scores = {s.category: s for s in build_category_scores(make_config(), metrics, None)}
    assert scores[C.PACE].score is None and scores[C.PACE].feedback == NOT_ENOUGH_SPEECH


def test_overall_score_uses_weights_and_skips_unscored() -> None:
    config = make_config(criteria=("pace", "filler_words", "grammar"), weights={"pace": 3.0})
    metrics = GOOD_METRICS.model_copy(update={"fillers_per_100_words": 6.0})
    categories = build_category_scores(config, metrics, None)
    # Weighted mean (10 * 3 + 5 * 1) / 4 = 8.75, reported to one decimal.
    assert overall_score(config, categories) == 8.8
    assert overall_score(make_config(criteria=("grammar",)), []) is None


def test_only_selected_criteria_are_reported() -> None:
    config = make_config(criteria=("pace", "grammar"))
    assert [s.category for s in build_category_scores(config, GOOD_METRICS, None)] == [
        C.PACE,
        C.GRAMMAR,
    ]
