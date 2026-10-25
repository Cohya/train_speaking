"""Rule-based interviewer: works without an LLM and serves as the LLM fallback."""

from __future__ import annotations

from train_speaking.interviewer.base import InterviewContext
from train_speaking.interviewer.planner import FINAL_QUESTION, fallback_plan
from train_speaking.interviewer.policies import (
    InterviewPhase,
    can_follow_up,
    is_repetitive,
    unused_topics,
)
from train_speaking.models.config import InterviewConfig, InterviewStyle
from train_speaking.models.interview import InterviewerAction, InterviewerTurn, InterviewPlan

SHORT_ANSWER_WORDS = 25

_FOLLOW_UPS = [
    "Could you give me a specific example of that?",
    "What was the outcome, and how did you measure it?",
    "What was your personal contribution, as opposed to the team's?",
    "Looking back, what would you do differently?",
]

_ACKNOWLEDGEMENT = {
    InterviewStyle.FRIENDLY: "Thanks, that's helpful.",
    InterviewStyle.NEUTRAL: "Thank you.",
    InterviewStyle.CHALLENGING: "Okay.",
}


class ScriptedInterviewer:
    async def plan(self, config: InterviewConfig) -> InterviewPlan:
        return fallback_plan(config)

    def greeting(self, ctx: InterviewContext) -> str:
        if ctx.config.practice_question:
            return "Let's practice this question."
        return (
            "Hello, and thank you for joining me today. I'll be interviewing you for the "
            f"{ctx.plan.role_title} position."
        )

    async def next_turn(self, ctx: InterviewContext) -> InterviewerTurn:
        return self.fallback_turn(ctx)

    def fallback_turn(self, ctx: InterviewContext) -> InterviewerTurn:
        asked = ctx.asked_questions()
        if not asked:
            first = ctx.plan.topics[0]
            return InterviewerTurn(
                action=InterviewerAction.NEXT_TOPIC,
                question=f"{self.greeting(ctx)} {first.opening_question}",
                topic_id=first.id,
                used_fallback=True,
            )
        # Practice mode asks exactly one question, so any answer ends the session.
        if ctx.phase is InterviewPhase.CLOSE or ctx.config.practice_question:
            return InterviewerTurn(
                action=InterviewerAction.CLOSE, question=self.closing(ctx), used_fallback=True
            )
        # With spoken acknowledgements enabled, "Thank you." is already said at turn end.
        ack = "" if ctx.config.acknowledgements else _ACKNOWLEDGEMENT[ctx.config.style]
        if ctx.phase is InterviewPhase.WRAP_UP:
            return InterviewerTurn(
                action=InterviewerAction.WRAP_UP,
                question=f"{ack} {FINAL_QUESTION}".strip(),
                topic_id="closing",
                used_fallback=True,
            )

        last_question = ctx.last_question()
        last_answer = ctx.last_answer()
        topic_id = last_question.topic_id if last_question else None
        answer_words = len(last_answer.text.split()) if last_answer else 0
        if answer_words < SHORT_ANSWER_WORDS and can_follow_up(ctx.turns, topic_id):
            for follow_up in _FOLLOW_UPS:
                if not is_repetitive(follow_up, asked):
                    return InterviewerTurn(
                        action=InterviewerAction.FOLLOW_UP,
                        question=follow_up,
                        topic_id=topic_id,
                        is_follow_up=True,
                        used_fallback=True,
                    )

        for topic in unused_topics(ctx.plan, ctx.turns):
            return InterviewerTurn(
                action=InterviewerAction.NEXT_TOPIC,
                question=f"{ack} {topic.opening_question}".strip(),
                topic_id=topic.id,
                used_fallback=True,
            )
        for topic in ctx.plan.topics:
            for backup in topic.backup_questions:
                if not is_repetitive(backup, asked):
                    return InterviewerTurn(
                        action=InterviewerAction.NEXT_TOPIC,
                        question=f"{ack} {backup}".strip(),
                        topic_id=topic.id,
                        used_fallback=True,
                    )
        return InterviewerTurn(
            action=InterviewerAction.WRAP_UP,
            question=f"{ack} {FINAL_QUESTION}".strip(),
            topic_id="closing",
            used_fallback=True,
        )

    def closing(self, ctx: InterviewContext) -> str:
        if ctx.config.practice_question:
            return "Thank you. I'll prepare feedback on your answer now."
        return (
            "Thank you for your time today. That concludes our interview. "
            "I'll now prepare your feedback."
        )
