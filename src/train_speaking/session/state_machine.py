"""Allowed session state transitions."""

from __future__ import annotations

from train_speaking.models.session import SessionState

S = SessionState

_ACTIVE_EXITS = frozenset({S.PAUSED, S.EVALUATING, S.ABORTED})

TRANSITIONS: dict[SessionState, frozenset[SessionState]] = {
    S.CONFIGURING: frozenset({S.READY, S.ABORTED}),
    S.READY: frozenset({S.PROCESSING, S.ABORTED}),
    S.PROCESSING: frozenset({S.INTERVIEWER_SPEAKING, S.WRAPPING_UP, S.RECOVERING, S.LISTENING})
    | _ACTIVE_EXITS,
    S.INTERVIEWER_SPEAKING: frozenset({S.LISTENING, S.USER_SPEAKING}) | _ACTIVE_EXITS,
    S.LISTENING: frozenset({S.USER_SPEAKING, S.PROCESSING}) | _ACTIVE_EXITS,
    S.USER_SPEAKING: frozenset({S.PROCESSING, S.LISTENING}) | _ACTIVE_EXITS,
    S.RECOVERING: frozenset({S.INTERVIEWER_SPEAKING, S.PROCESSING}) | _ACTIVE_EXITS,
    S.PAUSED: frozenset(
        {S.LISTENING, S.INTERVIEWER_SPEAKING, S.PROCESSING, S.EVALUATING, S.ABORTED}
    ),
    S.WRAPPING_UP: frozenset({S.EVALUATING, S.ABORTED}),
    S.EVALUATING: frozenset({S.COMPLETED, S.ABORTED}),
    S.COMPLETED: frozenset(),
    S.ABORTED: frozenset(),
}

INTERVIEW_ACTIVE_STATES = frozenset(
    {S.PROCESSING, S.INTERVIEWER_SPEAKING, S.LISTENING, S.USER_SPEAKING, S.RECOVERING}
)
TERMINAL_STATES = frozenset({S.COMPLETED, S.ABORTED})


class InvalidTransitionError(RuntimeError):
    def __init__(self, source: SessionState, target: SessionState) -> None:
        super().__init__(f"invalid session transition {source.value} -> {target.value}")
        self.source = source
        self.target = target


class SessionStateMachine:
    def __init__(self, initial: SessionState = S.CONFIGURING) -> None:
        self._state = initial
        self._paused_from: SessionState | None = None

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def paused_from(self) -> SessionState | None:
        """The state that was active when the session was paused."""
        return self._paused_from

    def can_transition(self, target: SessionState) -> bool:
        return target in TRANSITIONS[self._state]

    def transition(self, target: SessionState) -> None:
        if not self.can_transition(target):
            raise InvalidTransitionError(self._state, target)
        if target is S.PAUSED:
            self._paused_from = self._state
        elif self._state is S.PAUSED:
            self._paused_from = None
        self._state = target
