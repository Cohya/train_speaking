from __future__ import annotations

from tests.conftest import make_config
from train_speaking.interviewer.base import InterviewContext
from train_speaking.interviewer.planner import FINAL_QUESTION, fallback_plan, role_title_from
from train_speaking.interviewer.policies import InterviewPhase
from train_speaking.interviewer.scripted import ScriptedInterviewer
from train_speaking.models.config import InterviewConfig, InterviewFocus
from train_speaking.models.interview import InterviewerAction, InterviewerTurn, TopicKind
from train_speaking.models.transcript import Role, Turn

LONG_ANSWER = " ".join(["detail"] * 40)


def _context(
    config: InterviewConfig, turns: list[Turn], phase: InterviewPhase = InterviewPhase.MAIN
) -> InterviewContext:
    return InterviewContext(
        config=config, plan=fallback_plan(config), turns=turns, elapsed_s=60, phase=phase
    )


def _ask(turns: list[Turn], decision: InterviewerTurn, answer: str) -> None:
    turns.append(
        Turn(
            id=len(turns) + 1,
            role=Role.INTERVIEWER,
            text=decision.question,
            started_at=0,
            ended_at=0,
            topic_id=decision.topic_id,
            is_follow_up=decision.is_follow_up,
            is_final_question=decision.action is InterviewerAction.WRAP_UP,
        )
    )
    turns.append(
        Turn(id=len(turns) + 1, role=Role.CANDIDATE, text=answer, started_at=0, ended_at=0)
    )


def test_role_title_from_first_line() -> None:
    assert role_title_from("\n# Senior Backend Engineer\nDetails") == "Senior Backend Engineer"
    assert role_title_from("   ") == "this"


def test_fallback_plan_respects_focus() -> None:
    technical = fallback_plan(make_config(focus=InterviewFocus.TECHNICAL))
    assert technical.topics[0].kind is TopicKind.OPENING
    assert {t.kind for t in technical.topics[1:]} == {TopicKind.TECHNICAL}
    mixed = fallback_plan(make_config(focus=InterviewFocus.MIXED))
    assert [t.kind for t in mixed.topics[1:3]] == [TopicKind.TECHNICAL, TopicKind.BEHAVIORAL]


def test_practice_plan_has_single_question() -> None:
    plan = fallback_plan(make_config(practice_question="Tell me about yourself."))
    assert [t.opening_question for t in plan.topics] == ["Tell me about yourself."]


async def test_first_turn_greets_and_opens() -> None:
    config = make_config()
    turn = await ScriptedInterviewer().next_turn(_context(config, []))
    assert turn.topic_id == "opening"
    assert "Senior Backend Engineer" in turn.question


async def test_short_answers_get_limited_follow_ups_then_new_topic() -> None:
    config = make_config()
    interviewer = ScriptedInterviewer()
    turns: list[Turn] = []
    _ask(turns, await interviewer.next_turn(_context(config, turns)), "Short answer.")
    decisions = []
    for _ in range(3):
        decision = await interviewer.next_turn(_context(config, turns))
        decisions.append(decision)
        _ask(turns, decision, "Still short.")
    assert [d.is_follow_up for d in decisions] == [True, True, False]
    assert decisions[0].question != decisions[1].question


async def test_long_answer_moves_to_next_topic_without_repeating() -> None:
    config = make_config()
    interviewer = ScriptedInterviewer()
    turns: list[Turn] = []
    seen_topics = []
    for _ in range(12):
        decision = await interviewer.next_turn(_context(config, turns))
        if decision.action is InterviewerAction.WRAP_UP:
            break
        seen_topics.append(decision.topic_id)
        _ask(turns, decision, LONG_ANSWER)
    questions = [t.text for t in turns if t.role is Role.INTERVIEWER]
    assert len(questions) == len(set(questions))
    assert len(set(seen_topics)) == len(fallback_plan(config).topics)


async def test_wrap_up_and_close_phases() -> None:
    config = make_config()
    interviewer = ScriptedInterviewer()
    turns: list[Turn] = []
    _ask(turns, await interviewer.next_turn(_context(config, turns)), LONG_ANSWER)
    wrap = await interviewer.next_turn(_context(config, turns, InterviewPhase.WRAP_UP))
    assert wrap.action is InterviewerAction.WRAP_UP and FINAL_QUESTION in wrap.question
    close = await interviewer.next_turn(_context(config, turns, InterviewPhase.CLOSE))
    assert close.action is InterviewerAction.CLOSE
