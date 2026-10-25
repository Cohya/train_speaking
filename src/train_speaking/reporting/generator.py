"""Assembles the final report from scores, measurements, and judgments."""

from __future__ import annotations

from datetime import UTC, datetime

from train_speaking.evaluation.llm_judge import JudgeOutput
from train_speaking.evaluation.rubric import PACE_BAND_WPM
from train_speaking.models.config import EvaluationCategory as C
from train_speaking.models.report import (
    CategoryScore,
    Correction,
    PracticeComparison,
    PracticeDrill,
    QuestionBreakdown,
    Report,
    ScoreBasis,
    SessionMetricsSummary,
)
from train_speaking.models.session import Session

STRONG_SCORE = 8
WEAK_SCORE = 5
MAX_DRILLS = 6

METHODOLOGY = [
    "Measured scores (pace, pauses, filler words, lexical diversity) are computed in code from "
    "word timestamps produced by the speech recognizer.",
    "AI-judgment scores come from a language model reading the transcript; each cites quotes "
    "that were checked against your actual answers.",
    "Filler counts are a lower bound because speech recognizers often omit 'um' and 'uh'.",
    "Pronunciation is not scored from text or recognizer confidence; accent is never rated.",
    "The camera preview is not analyzed: no facial-expression, emotion, or personality "
    "inference is performed.",
]

_MEASURED_DRILLS: dict[C, PracticeDrill] = {
    C.FILLER_WORDS: PracticeDrill(
        title="Pause instead of filling",
        instructions="Answer one question aloud for 60 seconds. Whenever you feel an 'um' "
        "coming, close your mouth and pause silently instead.",
        target="Fewer than 2 um/uh per 100 words.",
    ),
    C.PACE: PracticeDrill(
        title="Find a steady pace",
        instructions="Read a short answer aloud with a timer, marking a breath after each "
        "sentence, then answer the same question without notes.",
        target=f"{PACE_BAND_WPM[0]:.0f}-{PACE_BAND_WPM[1]:.0f} words per minute, at most one "
        "pause over 2 seconds per minute.",
    ),
    C.FLUENCY: PracticeDrill(
        title="Plan before you speak",
        instructions="Before answering, take two seconds to choose a structure (situation, "
        "task, action, result) so long mid-answer pauses become planned ones.",
        target="At most two pauses over 1 second per minute.",
    ),
    C.VOCABULARY: PracticeDrill(
        title="Vary your wording",
        instructions="List three precise verbs for your key achievements (designed, reduced, "
        "negotiated) and use them in your next practice answers.",
        target="No phrase repeated more than twice per answer.",
    ),
}


def _breakdown(session: Session) -> list[QuestionBreakdown]:
    items = []
    for answer in session.candidate_turns():
        question = session.question_for(answer)
        if question is None:
            continue
        metrics = answer.metrics
        items.append(
            QuestionBreakdown(
                question_turn_id=question.id,
                question=question.text,
                answer_turn_id=answer.id,
                answer=answer.text,
                audio_file=answer.audio_file,
                metrics={
                    "words": metrics.word_count if metrics else None,
                    "wpm": metrics.wpm if metrics else None,
                    "pauses_over_2s": metrics.pause_count_gt_2s if metrics else None,
                    "response_latency_s": metrics.response_latency_s if metrics else None,
                },
            )
        )
    return items


def _measured_notes(categories: list[CategoryScore]) -> tuple[list[str], list[str]]:
    strengths, weaknesses = [], []
    for category in categories:
        if category.basis is not ScoreBasis.MEASURED or category.score is None:
            continue
        note = f"{category.label}: {category.feedback}"
        if category.score >= STRONG_SCORE:
            strengths.append(note)
        elif category.score <= WEAK_SCORE:
            weaknesses.append(note)
    return strengths, weaknesses


def _summary(session: Session, metrics: SessionMetricsSummary, judge: JudgeOutput | None) -> str:
    if judge is not None and judge.summary:
        return judge.summary
    minutes = metrics.total_speaking_s / 60
    text = f"You answered {metrics.answer_count} question(s) and spoke for {minutes:.1f} minutes."
    if judge is None:
        text += (
            " Language-level categories were not assessed because no AI evaluator was "
            "available; measured categories are shown."
        )
    return text


def build_report(
    session: Session,
    categories: list[CategoryScore],
    metrics: SessionMetricsSummary,
    overall: float | None,
    judge: JudgeOutput | None,
    *,
    flagged_evidence: int = 0,
    judge_error: str | None = None,
    comparison: PracticeComparison | None = None,
) -> Report:
    measured_strengths, measured_weaknesses = _measured_notes(categories)
    strengths = (judge.strengths if judge else []) + measured_strengths
    weaknesses = (judge.weaknesses if judge else []) + measured_weaknesses

    drills = [PracticeDrill(**d.model_dump()) for d in judge.practice_plan] if judge else []
    titles = {d.title.lower() for d in drills}
    for category in categories:
        drill = _MEASURED_DRILLS.get(category.category)
        weak = category.score is not None and category.score <= WEAK_SCORE + 1
        if drill is not None and weak and drill.title.lower() not in titles:
            drills.append(drill)
            titles.add(drill.title.lower())

    corrections = (
        [Correction(**c.model_dump()) for c in judge.corrections] if judge is not None else []
    )
    methodology = list(METHODOLOGY)
    if judge_error:
        methodology.append(f"AI evaluation was unavailable ({judge_error}).")
    if session.errors:
        methodology.append(f"{len(session.errors)} technical issue(s) occurred and were handled.")

    return Report(
        session_id=session.id,
        generated_at=datetime.now(UTC),
        role_title=session.plan.role_title if session.plan else "Interview",
        overall_score=overall,
        summary=_summary(session, metrics, judge),
        categories=categories,
        metrics=metrics,
        questions=_breakdown(session),
        strengths=strengths,
        weaknesses=weaknesses,
        corrections=corrections,
        practice_plan=drills[:MAX_DRILLS],
        methodology=methodology,
        flagged_evidence_count=flagged_evidence,
        practice_comparison=comparison,
    )
