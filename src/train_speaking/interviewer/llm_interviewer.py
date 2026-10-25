"""LLM-driven interviewer with deterministic policy enforcement and a scripted fallback."""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel, Field

from train_speaking.interviewer.base import InterviewContext
from train_speaking.interviewer.planner import fallback_plan
from train_speaking.interviewer.policies import (
    MAX_QUESTION_SENTENCES,
    InterviewPhase,
    can_follow_up,
    follow_up_count,
    is_repetitive,
    unused_topics,
)
from train_speaking.interviewer.prompts import plan_messages, turn_messages
from train_speaking.interviewer.scripted import ScriptedInterviewer
from train_speaking.models.config import InterviewConfig
from train_speaking.models.interview import (
    InterviewerAction,
    InterviewerTurn,
    InterviewPlan,
    PlannedTopic,
    TopicKind,
)
from train_speaking.providers.llm.base import LLMProvider, UsageTracker, complete_structured
from train_speaking.voice.sentence_splitter import limit_sentences

logger = logging.getLogger(__name__)

MIN_PLAN_TOKENS = 1200
_SLUG = re.compile(r"[^a-z0-9_\-]+")


class _TopicDraft(BaseModel):
    id: str
    title: str
    kind: Literal["opening", "technical", "behavioral", "communication"]
    minutes: float = 1.5
    opening_question: str
    backup_questions: list[str] = Field(default_factory=list)


class _PlanDraft(BaseModel):
    role_title: str
    topics: list[_TopicDraft] = Field(min_length=1, max_length=10)


class _Decision(BaseModel):
    action: Literal["follow_up", "next_topic", "wrap_up"]
    topic_id: str = ""
    question: str = Field(min_length=3)


def normalize_plan(draft: _PlanDraft, config: InterviewConfig) -> InterviewPlan:
    """Unique slug ids, an opening topic first, and minutes scaled to the duration."""
    topics: list[PlannedTopic] = []
    seen: set[str] = set()
    for index, topic in enumerate(draft.topics):
        slug = _SLUG.sub("-", topic.id.lower()).strip("-") or f"t{index}"
        if slug in seen:
            slug = f"{slug}-{index}"
        seen.add(slug)
        topics.append(
            PlannedTopic(
                id=slug,
                title=topic.title.strip()[:80] or f"Topic {index + 1}",
                kind=TopicKind(topic.kind),
                minutes=max(0.5, topic.minutes),
                opening_question=limit_sentences(topic.opening_question, MAX_QUESTION_SENTENCES),
                backup_questions=[
                    limit_sentences(q, MAX_QUESTION_SENTENCES) for q in topic.backup_questions[:2]
                ],
            )
        )
    if topics[0].kind is not TopicKind.OPENING:
        topics.insert(0, fallback_plan(config).topics[0])
    total = sum(t.minutes for t in topics)
    budget = max(1.0, config.duration_min - 1)
    if total > budget:
        topics = [
            t.model_copy(update={"minutes": round(t.minutes * budget / total, 1)}) for t in topics
        ]
    return InterviewPlan(role_title=draft.role_title.strip()[:60] or "this", topics=topics)


class LLMInterviewer:
    def __init__(
        self,
        llm: LLMProvider,
        *,
        tracker: UsageTracker,
        max_tokens: int,
        fallback: ScriptedInterviewer | None = None,
    ) -> None:
        self._llm = llm
        self._tracker = tracker
        self._max_tokens = max_tokens
        self._fallback = fallback or ScriptedInterviewer()

    async def plan(self, config: InterviewConfig) -> InterviewPlan:
        if config.practice_question:
            return fallback_plan(config)
        draft = await complete_structured(
            self._llm,
            plan_messages(config),
            _PlanDraft,
            max_tokens=max(MIN_PLAN_TOKENS, self._max_tokens),
            temperature=0.5,
            tracker=self._tracker,
        )
        return normalize_plan(draft, config)

    def closing(self, ctx: InterviewContext) -> str:
        return self._fallback.closing(ctx)

    async def next_turn(self, ctx: InterviewContext) -> InterviewerTurn:
        if (
            not ctx.asked_questions()
            or ctx.phase is InterviewPhase.CLOSE
            or ctx.config.practice_question
        ):
            return self._fallback.fallback_turn(ctx)
        asked = ctx.asked_questions()
        retry_note: str | None = None
        for _attempt in range(2):
            decision = await self._decide(ctx, retry_note)
            turn = self._enforce(decision, ctx)
            if turn is None:
                break
            if not is_repetitive(turn.question, asked):
                return turn
            retry_note = (
                f'Your question "{turn.question}" repeats an earlier question. Ask something '
                "different."
            )
        logger.info("llm interviewer fell back to scripted question")
        return self._fallback.fallback_turn(ctx)

    async def _decide(self, ctx: InterviewContext, retry_note: str | None) -> _Decision:
        last_question = ctx.last_question()
        current_topic = last_question.topic_id if last_question else None
        covered = {t.topic_id for t in ctx.turns if t.topic_id}
        messages = turn_messages(
            ctx.config,
            ctx.plan,
            ctx.turns,
            elapsed_s=ctx.elapsed_s,
            phase=ctx.phase,
            current_topic_id=current_topic,
            follow_ups_on_topic=follow_up_count(ctx.turns, current_topic) if current_topic else 0,
            covered_topic_ids={c for c in covered if c is not None},
            retry_note=retry_note,
        )
        return await complete_structured(
            self._llm, messages, _Decision, max_tokens=self._max_tokens, tracker=self._tracker
        )

    def _enforce(self, decision: _Decision, ctx: InterviewContext) -> InterviewerTurn | None:
        """Apply the policies the LLM may ignore. Returns None to request the fallback."""
        question = limit_sentences(decision.question.strip().strip('"'), MAX_QUESTION_SENTENCES)
        if not question:
            return None
        if ctx.phase is InterviewPhase.WRAP_UP:
            return InterviewerTurn(
                action=InterviewerAction.WRAP_UP, question=question, topic_id="closing"
            )
        last_question = ctx.last_question()
        current_topic = last_question.topic_id if last_question else None
        if decision.action == "follow_up":
            if not can_follow_up(ctx.turns, current_topic):
                return None
            return InterviewerTurn(
                action=InterviewerAction.FOLLOW_UP,
                question=question,
                topic_id=current_topic,
                is_follow_up=True,
            )
        if decision.action == "wrap_up":
            return InterviewerTurn(
                action=InterviewerAction.WRAP_UP, question=question, topic_id="closing"
            )
        known = {t.id for t in ctx.plan.topics}
        topic_id = decision.topic_id if decision.topic_id in known else None
        if topic_id is None:
            remaining = unused_topics(ctx.plan, ctx.turns)
            topic_id = remaining[0].id if remaining else "general"
        return InterviewerTurn(
            action=InterviewerAction.NEXT_TOPIC, question=question, topic_id=topic_id
        )
