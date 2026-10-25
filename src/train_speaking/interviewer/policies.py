"""Deterministic interview-progression rules, enforced in code rather than in prompts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher
from enum import StrEnum

from train_speaking.models.interview import InterviewPlan, PlannedTopic
from train_speaking.models.transcript import Role, Turn

WRAP_UP_LEAD_S = 90.0
HARD_STOP_GRACE_S = 120.0
MAX_FOLLOW_UPS_PER_TOPIC = 2
MAX_QUESTION_SENTENCES = 3
REPETITION_SIMILARITY = 0.8
NO_ANSWER_TIMEOUT_S = 20.0
MAX_NO_ANSWER_NUDGES = 2

_NON_WORD = re.compile(r"[^a-z0-9 ]+")


class InterviewPhase(StrEnum):
    MAIN = "main"
    WRAP_UP = "wrap_up"
    CLOSE = "close"


def phase_for(turns: Sequence[Turn], elapsed_s: float, duration_s: float) -> InterviewPhase:
    if elapsed_s >= duration_s or any(t.is_final_question for t in turns):
        return InterviewPhase.CLOSE
    if duration_s - elapsed_s <= WRAP_UP_LEAD_S:
        return InterviewPhase.WRAP_UP
    return InterviewPhase.MAIN


def is_hard_stop(elapsed_s: float, duration_s: float) -> bool:
    return elapsed_s >= duration_s + HARD_STOP_GRACE_S


def _interviewer_questions(turns: Sequence[Turn]) -> list[Turn]:
    return [t for t in turns if t.role is Role.INTERVIEWER and t.topic_id]


def follow_up_count(turns: Sequence[Turn], topic_id: str) -> int:
    """Consecutive follow-ups asked on `topic_id` since its last main question."""
    count = 0
    for turn in reversed(_interviewer_questions(turns)):
        if turn.topic_id != topic_id or not turn.is_follow_up:
            break
        count += 1
    return count


def can_follow_up(turns: Sequence[Turn], topic_id: str | None) -> bool:
    return topic_id is not None and follow_up_count(turns, topic_id) < MAX_FOLLOW_UPS_PER_TOPIC


def unused_topics(plan: InterviewPlan, turns: Sequence[Turn]) -> list[PlannedTopic]:
    used = {t.topic_id for t in _interviewer_questions(turns)}
    return [topic for topic in plan.topics if topic.id not in used]


def _normalize(text: str) -> str:
    return " ".join(_NON_WORD.sub(" ", text.lower()).split())


def is_repetitive(question: str, previous: Sequence[str]) -> bool:
    candidate = _normalize(question)
    return any(
        SequenceMatcher(None, candidate, _normalize(earlier)).ratio() >= REPETITION_SIMILARITY
        for earlier in previous
    )
