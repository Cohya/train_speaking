"""Interviewer interface and the context passed to it on each turn."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from train_speaking.interviewer.policies import InterviewPhase
from train_speaking.models.config import InterviewConfig
from train_speaking.models.interview import InterviewerTurn, InterviewPlan
from train_speaking.models.transcript import Role, Turn


@dataclass(frozen=True)
class InterviewContext:
    config: InterviewConfig
    plan: InterviewPlan
    turns: Sequence[Turn]
    elapsed_s: float
    phase: InterviewPhase

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.config.duration_s - self.elapsed_s)

    def asked_questions(self) -> list[str]:
        return [t.text for t in self.turns if t.role is Role.INTERVIEWER and t.topic_id]

    def last_question(self) -> Turn | None:
        questions = [t for t in self.turns if t.role is Role.INTERVIEWER and t.topic_id]
        return questions[-1] if questions else None

    def last_answer(self) -> Turn | None:
        answers = [t for t in self.turns if t.role is Role.CANDIDATE]
        return answers[-1] if answers else None


class Interviewer(Protocol):
    async def plan(self, config: InterviewConfig) -> InterviewPlan: ...

    async def next_turn(self, ctx: InterviewContext) -> InterviewerTurn: ...

    def closing(self, ctx: InterviewContext) -> str: ...
