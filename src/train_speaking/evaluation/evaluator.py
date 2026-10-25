"""Speaking evaluation agent: measurements, optional LLM judgment, scoring, and report."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

from train_speaking.evaluation.aggregate import summarize_metrics
from train_speaking.evaluation.evidence_validation import validate_judge_output
from train_speaking.evaluation.llm_judge import JudgeOutput, run_judge
from train_speaking.evaluation.pronunciation import PronunciationAssessor, PronunciationResult
from train_speaking.evaluation.rubric import JUDGED_CATEGORIES
from train_speaking.evaluation.scoring import build_category_scores, overall_score
from train_speaking.models.audio import AudioClip
from train_speaking.models.report import PracticeComparison, Report
from train_speaking.models.session import Session
from train_speaking.models.transcript import AnswerMetrics, Turn
from train_speaking.providers.llm.base import LLMError, LLMProvider, UsageTracker
from train_speaking.reporting.generator import build_report

logger = logging.getLogger(__name__)

SessionLookup = Callable[[str], Session | None]
ClipLoader = Callable[[Session], Mapping[int, AudioClip]]


def _metrics_dict(metrics: AnswerMetrics | None) -> dict[str, float | int | None]:
    if metrics is None:
        return {}
    words = metrics.word_count
    return {
        "words": words,
        "duration_s": metrics.duration_s,
        "wpm": metrics.wpm,
        "um_uh_per_100_words": (
            round(100 * metrics.fillers.definite_total / words, 2) if words else None
        ),
        "pauses_over_2s": metrics.pause_count_gt_2s,
    }


class SpeakingEvaluator:
    def __init__(
        self,
        llm: LLMProvider | None,
        *,
        tracker: UsageTracker | None = None,
        max_tokens: int = 2500,
        pronunciation: PronunciationAssessor | None = None,
        clip_loader: ClipLoader | None = None,
        session_lookup: SessionLookup | None = None,
    ) -> None:
        self._llm = llm
        self._tracker = tracker
        self._max_tokens = max_tokens
        self._pronunciation = pronunciation
        self._clip_loader = clip_loader
        self._session_lookup = session_lookup

    async def evaluate(self, session: Session) -> Report:
        metrics = summarize_metrics(session)
        judge: JudgeOutput | None = None
        flagged = 0
        judge_error: str | None = None
        judged = [c for c in session.config.criteria if c in JUDGED_CATEGORIES]
        if self._llm is not None and judged and session.candidate_turns():
            try:
                raw = await run_judge(
                    self._llm,
                    session,
                    judged,
                    metrics,
                    max_tokens=self._max_tokens,
                    tracker=self._tracker,
                )
                judge, flagged = validate_judge_output(raw, session)
            except LLMError as exc:
                judge_error = type(exc).__name__
                logger.warning("llm judge failed", extra={"error": judge_error})

        pronunciation = await self._assess_pronunciation(session)
        categories = build_category_scores(session.config, metrics, judge, pronunciation)
        return build_report(
            session,
            categories,
            metrics,
            overall_score(session.config, categories),
            judge,
            flagged_evidence=flagged,
            judge_error=judge_error,
            comparison=self._practice_comparison(session),
        )

    async def _assess_pronunciation(self, session: Session) -> PronunciationResult | None:
        if self._pronunciation is None or self._clip_loader is None:
            return None
        clips = self._clip_loader(session)
        if not clips:
            return None
        return await self._pronunciation.assess(session.candidate_turns(), clips)

    def _practice_comparison(self, session: Session) -> PracticeComparison | None:
        if (
            session.parent_session_id is None
            or session.practice_turn_id is None
            or self._session_lookup is None
        ):
            return None
        parent = self._session_lookup(session.parent_session_id)
        if parent is None:
            return None
        original: Turn | None = next(
            (t for t in parent.turns if t.id == session.practice_turn_id), None
        )
        answers = session.candidate_turns()
        if original is None or not answers:
            return None
        new = answers[0]
        return PracticeComparison(
            original_turn_id=original.id,
            original_answer=original.text,
            new_answer=new.text,
            original_metrics=_metrics_dict(original.metrics),
            new_metrics=_metrics_dict(new.metrics),
        )
