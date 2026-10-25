from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from tests.conftest import make_config
from train_speaking.evaluation.evaluator import SpeakingEvaluator
from train_speaking.evaluation.evidence_validation import quote_in_text, validate_judge_output
from train_speaking.evaluation.llm_judge import JudgeOutput
from train_speaking.evaluation.pronunciation import PronunciationResult
from train_speaking.interviewer.planner import fallback_plan
from train_speaking.models.audio import AudioClip
from train_speaking.models.config import EvaluationCategory as C
from train_speaking.models.report import ScoreBasis
from train_speaking.models.session import Session, SessionState
from train_speaking.models.transcript import AnswerMetrics, FillerCounts, Role, Turn, WordTiming
from train_speaking.providers.llm.base import LLMError
from train_speaking.providers.llm.fake import FakeLLM
from train_speaking.reporting.markdown import render_markdown

ANSWER_1 = (
    "Um, I have, uh, led the backend team for three years and we was migrating to async "
    "services. At the end we reduced latency by forty percent."
)
ANSWER_2 = "I usually talk with the colleague first and then we decide together on facts."


def _answer(turn_id: int, text: str, start: float) -> Turn:
    words = [
        WordTiming(word=w, start=start + i * 0.4, end=start + i * 0.4 + 0.3, probability=0.9)
        for i, w in enumerate(text.split())
    ]
    n = len(words)
    return Turn(
        id=turn_id,
        role=Role.CANDIDATE,
        text=text,
        started_at=start,
        ended_at=start + n * 0.4,
        words=words,
        metrics=AnswerMetrics(
            duration_s=n * 0.4,
            word_count=n,
            wpm=150,
            pause_count_gt_1s=1,
            pause_count_gt_2s=0,
            longest_pause_s=1.2,
            response_latency_s=0.8,
            fillers=FillerCounts(definite={"um": 1, "uh": 1}, total_words=n),
            mean_word_probability=0.9,
        ),
    )


def _session(**config_overrides: object) -> Session:
    config = make_config(**config_overrides)
    turns = [
        Turn(id=1, role=Role.INTERVIEWER, text="Tell me about yourself.", started_at=0,
             ended_at=2, topic_id="opening"),
        _answer(2, ANSWER_1, 3.0),
        Turn(id=3, role=Role.INTERVIEWER, text="How do you handle disagreement?", started_at=20,
             ended_at=22, topic_id="t2"),
        _answer(4, ANSWER_2, 23.0),
    ]  # fmt: skip
    session = Session(
        id="eval-session-1",
        config=config,
        plan=fallback_plan(config),
        state=SessionState.EVALUATING,
        turns=turns,
    )
    # Pad speaking time so measured categories have enough data.
    for turn in session.candidate_turns():
        assert turn.metrics is not None
        turn.metrics.duration_s = 30
    return session


def _judge_json(**overrides: object) -> str:
    payload: dict[str, object] = {
        "summary": "Clear and relevant answers with some grammar slips.",
        "categories": [
            {
                "category": "grammar",
                "score": 6,
                "evidence": [
                    {"turn_id": 2, "quote": "we was migrating to async services"},
                    {"turn_id": 2, "quote": "this sentence was never said"},
                ],
                "feedback": "Watch subject-verb agreement.",
            },
            {"category": "structure", "score": 8, "evidence": [], "feedback": "Good results."},
        ],
        "strengths": ["Quantified impact (forty percent)."],
        "weaknesses": ["Subject-verb agreement."],
        "corrections": [
            {
                "turn_id": 4,
                "original": "we was migrating to async services",
                "corrected": "we were migrating to async services",
                "natural": "we were moving our services to an asynchronous design",
            },
            {
                "turn_id": 2,
                "original": "an invented sentence",
                "corrected": "x",
                "natural": "y",
            },
        ],
        "practice_plan": [
            {"title": "STAR drill", "instructions": "Use STAR.", "target": "3 stories."}
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("we was migrating to async services", True),
        ("We was migrating, to async services!", True),
        ("we were migrating to async services", True),
        ("we was migrating to asynchronous microservices now", False),
        ("", False),
    ],
)
def test_quote_matching(quote: str, expected: bool) -> None:
    assert quote_in_text(quote, ANSWER_1) is expected


def test_validation_drops_unverifiable_quotes_and_fixes_turn_ids() -> None:
    session = _session()
    output = JudgeOutput.model_validate_json(_judge_json())
    validated, dropped = validate_judge_output(output, session)
    assert dropped == 2
    assert [e.quote for e in validated.categories[0].evidence] == [
        "we was migrating to async services"
    ]
    assert [c.turn_id for c in validated.corrections] == [2]


async def test_evaluator_with_llm_builds_full_report() -> None:
    llm = FakeLLM([_judge_json()])
    report = await SpeakingEvaluator(llm).evaluate(_session())
    categories = {c.category: c for c in report.categories}
    assert categories[C.GRAMMAR].score == 6
    assert categories[C.GRAMMAR].basis is ScoreBasis.MODEL_JUDGMENT
    assert categories[C.FILLER_WORDS].basis is ScoreBasis.MEASURED
    assert categories[C.PRONUNCIATION].basis is ScoreBasis.NOT_ASSESSED
    assert report.flagged_evidence_count == 2
    assert report.corrections[0].corrected == "we were migrating to async services"
    assert report.overall_score is not None
    assert report.summary.startswith("Clear and relevant")
    assert [q.question_turn_id for q in report.questions] == [1, 3]
    assert any(d.title == "STAR drill" for d in report.practice_plan)
    prompt = llm.calls[0][1].content
    assert "[turn 2] Candidate: <answer>" in prompt
    assert (
        "pronunciation" not in prompt.split("Return a JSON object")[1].split("categories")[1][:200]
    )


async def test_evaluator_survives_llm_failure() -> None:
    llm = FakeLLM([LLMError("down")])
    report = await SpeakingEvaluator(llm).evaluate(_session())
    assert any("AI evaluation was unavailable" in m for m in report.methodology)
    assert {c.category: c for c in report.categories}[C.GRAMMAR].score is None


async def test_evaluator_without_llm_uses_measurements_and_drills() -> None:
    session = _session()
    for turn in session.candidate_turns():
        assert turn.metrics is not None
        turn.metrics.fillers = FillerCounts(definite={"um": 8}, total_words=turn.metrics.word_count)
    report = await SpeakingEvaluator(None).evaluate(session)
    fillers = {c.category: c for c in report.categories}[C.FILLER_WORDS]
    assert fillers.score is not None and fillers.score <= 5
    assert any(d.title == "Pause instead of filling" for d in report.practice_plan)
    assert any("Filler-word frequency" in w for w in report.weaknesses)
    assert "not assessed" in report.summary


async def test_practice_comparison_uses_parent_session() -> None:
    parent = _session()
    practice_config = make_config(practice_question="Tell me about yourself.", duration_min=3)
    practice = Session(
        id="practice-session-1",
        config=practice_config,
        parent_session_id=parent.id,
        practice_turn_id=2,
        turns=[
            Turn(id=1, role=Role.INTERVIEWER, text="Tell me about yourself.", started_at=0,
                 ended_at=1, topic_id="practice"),
            _answer(2, "I lead a backend team and cut latency by forty percent.", 2.0),
        ],
    )  # fmt: skip
    evaluator = SpeakingEvaluator(
        None, session_lookup=lambda sid: parent if sid == parent.id else None
    )
    report = await evaluator.evaluate(practice)
    assert report.practice_comparison is not None
    assert report.practice_comparison.original_answer == ANSWER_1
    assert report.practice_comparison.new_metrics["words"] == 11


class _FakeAssessor:
    def __init__(self) -> None:
        self.seen: list[int] = []

    async def assess(
        self, answers: Sequence[Turn], audio: Mapping[int, AudioClip]
    ) -> PronunciationResult:
        self.seen = sorted(audio)
        return PronunciationResult(score=7, feedback="Intelligible throughout.")


async def test_pronunciation_plugin_scores_only_with_audio() -> None:
    assessor = _FakeAssessor()
    clip = AudioClip(np.zeros(1600, dtype=np.float32), 16_000)
    with_audio = SpeakingEvaluator(None, pronunciation=assessor, clip_loader=lambda s: {2: clip})
    report = await with_audio.evaluate(_session())
    pronunciation = {c.category: c for c in report.categories}[C.PRONUNCIATION]
    assert pronunciation.score == 7 and pronunciation.basis is ScoreBasis.MEASURED
    assert assessor.seen == [2]

    without_audio = SpeakingEvaluator(None, pronunciation=assessor, clip_loader=lambda s: {})
    report = await without_audio.evaluate(_session())
    pronunciation = {c.category: c for c in report.categories}[C.PRONUNCIATION]
    assert pronunciation.basis is ScoreBasis.NOT_ASSESSED


async def test_markdown_export_contains_key_sections() -> None:
    report = await SpeakingEvaluator(FakeLLM([_judge_json()])).evaluate(_session())
    markdown = render_markdown(report)
    for heading in (
        "# Interview report",
        "## Category scores",
        "## Measured metrics",
        "## Example corrections",
        "## Practice plan",
        "## Methodology and limitations",
    ):
        assert heading in markdown
    assert "(AI judgment)" in markdown and "(Measured)" in markdown
