from __future__ import annotations

import itertools

import pytest

from train_speaking.models.session import SessionState as S
from train_speaking.session.state_machine import (
    TERMINAL_STATES,
    TRANSITIONS,
    InvalidTransitionError,
    SessionStateMachine,
)


def test_every_state_has_a_transition_entry() -> None:
    assert set(TRANSITIONS) == set(S)


@pytest.mark.parametrize(("source", "target"), list(itertools.product(S, S)))
def test_transition_table_is_enforced(source: S, target: S) -> None:
    machine = SessionStateMachine(source)
    if target in TRANSITIONS[source]:
        machine.transition(target)
        assert machine.state is target
    else:
        with pytest.raises(InvalidTransitionError):
            machine.transition(target)
        assert machine.state is source


def test_terminal_states_have_no_exits() -> None:
    for state in TERMINAL_STATES:
        assert TRANSITIONS[state] == frozenset()


def test_happy_path_through_an_interview() -> None:
    machine = SessionStateMachine()
    for target in (
        S.READY,
        S.PROCESSING,
        S.INTERVIEWER_SPEAKING,
        S.LISTENING,
        S.USER_SPEAKING,
        S.PROCESSING,
        S.WRAPPING_UP,
        S.EVALUATING,
        S.COMPLETED,
    ):
        machine.transition(target)
    assert machine.state is S.COMPLETED


def test_pause_remembers_the_interrupted_state() -> None:
    machine = SessionStateMachine(S.USER_SPEAKING)
    machine.transition(S.PAUSED)
    assert machine.paused_from is S.USER_SPEAKING
    machine.transition(S.INTERVIEWER_SPEAKING)
    assert machine.paused_from is None


def test_end_is_possible_from_every_active_state() -> None:
    for state in (S.PROCESSING, S.INTERVIEWER_SPEAKING, S.LISTENING, S.USER_SPEAKING, S.PAUSED):
        assert S.EVALUATING in TRANSITIONS[state]
