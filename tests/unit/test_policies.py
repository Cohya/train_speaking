from __future__ import annotations

import pytest

from tests.conftest import make_config
from train_speaking.interviewer.planner import fallback_plan
from train_speaking.interviewer.policies import (
    HARD_STOP_GRACE_S,
    MAX_FOLLOW_UPS_PER_TOPIC,
    WRAP_UP_LEAD_S,
    InterviewPhase,
    can_follow_up,
    follow_up_count,
    is_hard_stop,
    is_repetitive,
    phase_for,
    unused_topics,
)
from train_speaking.models.transcript import Role, Turn


def _q(turn_id: int, topic: str, follow_up: bool = False, final: bool = False) -> Turn:
    return Turn(
        id=turn_id,
        role=Role.INTERVIEWER,
        text=f"question {turn_id}",
        started_at=0,
        ended_at=0,
        topic_id=topic,
        is_follow_up=follow_up,
        is_final_question=final,
    )


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [
        (0, InterviewPhase.MAIN),
        (300 - WRAP_UP_LEAD_S - 1, InterviewPhase.MAIN),
        (300 - WRAP_UP_LEAD_S, InterviewPhase.WRAP_UP),
        (299, InterviewPhase.WRAP_UP),
        (300, InterviewPhase.CLOSE),
    ],
)
def test_phase_by_time(elapsed: float, expected: InterviewPhase) -> None:
    assert phase_for([], elapsed, 300) is expected


def test_final_question_forces_close() -> None:
    assert phase_for([_q(1, "closing", final=True)], 10, 300) is InterviewPhase.CLOSE


def test_hard_stop() -> None:
    assert not is_hard_stop(300 + HARD_STOP_GRACE_S - 1, 300)
    assert is_hard_stop(300 + HARD_STOP_GRACE_S, 300)


def test_follow_up_counting_resets_on_new_main_question() -> None:
    turns = [_q(1, "a"), _q(2, "a", True), _q(3, "a", True)]
    assert follow_up_count(turns, "a") == 2
    assert not can_follow_up(turns, "a")
    turns.append(_q(4, "b"))
    assert follow_up_count(turns, "b") == 0
    assert can_follow_up(turns, "b")
    assert not can_follow_up(turns, None)
    assert MAX_FOLLOW_UPS_PER_TOPIC == 2


def test_unused_topics() -> None:
    plan = fallback_plan(make_config())
    turns = [_q(1, plan.topics[0].id), _q(2, plan.topics[1].id)]
    assert [t.id for t in unused_topics(plan, turns)] == [t.id for t in plan.topics[2:]]


@pytest.mark.parametrize(
    ("question", "previous", "expected"),
    [
        ("Tell me about a conflict.", ["Tell me about a conflict!"], True),
        ("tell me about a CONFLICT", ["Tell me about a conflict."], True),
        ("Tell me about a conflict.", ["How do you design APIs?"], False),
        ("Anything", [], False),
    ],
)
def test_repetition_guard(question: str, previous: list[str], expected: bool) -> None:
    assert is_repetitive(question, previous) is expected
