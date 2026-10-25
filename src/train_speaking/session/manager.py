"""In-memory registry of live sessions, keyed by an unguessable id plus an access token."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass

from train_speaking.models.config import InterviewConfig
from train_speaking.models.session import Session, SessionState
from train_speaking.session.orchestrator import InterviewOrchestrator
from train_speaking.session.state_machine import TERMINAL_STATES

MAX_LIVE_SESSIONS = 8

OrchestratorFactory = Callable[[Session], InterviewOrchestrator]


@dataclass
class SessionRuntime:
    session: Session
    orchestrator: InterviewOrchestrator
    token: str


class SessionLimitError(RuntimeError):
    pass


class SessionManager:
    def __init__(self, factory: OrchestratorFactory, max_sessions: int = MAX_LIVE_SESSIONS) -> None:
        self._factory = factory
        self._max = max_sessions
        self._sessions: dict[str, SessionRuntime] = {}

    def create(
        self,
        config: InterviewConfig,
        *,
        parent_session_id: str | None = None,
        practice_turn_id: int | None = None,
    ) -> SessionRuntime:
        self._evict_finished()
        if len(self._sessions) >= self._max:
            raise SessionLimitError("too many live sessions; finish or end one first")
        session = Session(
            id=secrets.token_urlsafe(12),
            config=config,
            state=SessionState.READY,
            parent_session_id=parent_session_id,
            practice_turn_id=practice_turn_id,
        )
        runtime = SessionRuntime(session, self._factory(session), secrets.token_urlsafe(24))
        self._sessions[session.id] = runtime
        return runtime

    def get(self, session_id: str) -> SessionRuntime | None:
        return self._sessions.get(session_id)

    def get_all(self) -> list[SessionRuntime]:
        return list(self._sessions.values())

    def verify(self, session_id: str, token: str) -> SessionRuntime | None:
        runtime = self._sessions.get(session_id)
        if runtime is None or not secrets.compare_digest(runtime.token, token):
            return None
        return runtime

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _evict_finished(self) -> None:
        if len(self._sessions) < self._max:
            return
        finished = [
            sid for sid, rt in self._sessions.items() if rt.session.state in TERMINAL_STATES
        ]
        for sid in finished[: len(self._sessions) - self._max + 1]:
            del self._sessions[sid]
