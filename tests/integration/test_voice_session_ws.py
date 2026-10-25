"""Full interview over `/ws/audio` with fake speech providers and a scripted interviewer."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from tests.conftest import make_config
from tests.helpers import make_orchestrator, silence_pcm, speech_pcm
from train_speaking.models.session import Session, SessionState
from train_speaking.providers.stt.fake import FakeSTT
from train_speaking.session.manager import SessionManager, SessionRuntime
from train_speaking.session.orchestrator import InterviewOrchestrator
from train_speaking.voice.protocol import decode_audio_chunk
from train_speaking.voice.ws_endpoint import create_ws_router

Message = dict[str, Any]


def _receive(ws: WebSocketTestSession) -> Message | bytes:
    message = ws.receive()
    if message.get("bytes") is not None:
        return bytes(message["bytes"])
    return dict(json.loads(message["text"]))


def receive_until(
    ws: WebSocketTestSession, predicate: Callable[[Message], bool], limit: int = 200
) -> tuple[Message, list[Message | bytes]]:
    seen: list[Message | bytes] = []
    for _ in range(limit):
        item = _receive(ws)
        seen.append(item)
        if isinstance(item, dict) and predicate(item):
            return item, seen
    raise AssertionError("expected message not received")


def finish_utterance(ws: WebSocketTestSession) -> list[Message | bytes]:
    end, seen = receive_until(ws, lambda m: m["type"] == "utterance_end")
    ws.send_text(json.dumps({"type": "playback_started", "utterance_id": end["utterance_id"]}))
    ws.send_text(json.dumps({"type": "playback_finished", "utterance_id": end["utterance_id"]}))
    return seen


def wait_state(ws: WebSocketTestSession, state: SessionState) -> None:
    receive_until(ws, lambda m: m["type"] == "state" and m["state"] == state.value)


@pytest.fixture
def stt() -> FakeSTT:
    return FakeSTT(
        [
            "I have led backend teams and built event pipelines in Python for years.",
            "We cut latency by half by moving work to async queues.",
        ]
    )


@pytest.fixture
def client_and_runtime(stt: FakeSTT) -> tuple[TestClient, SessionRuntime]:
    def factory(session: Session) -> InterviewOrchestrator:
        orchestrator = make_orchestrator(session.config, stt=stt, session_id=session.id)
        orchestrator.session = session
        return orchestrator

    manager = SessionManager(factory)
    runtime = manager.create(make_config())
    app = FastAPI()
    app.include_router(create_ws_router(manager))
    return TestClient(app), runtime


def test_invalid_token_is_rejected(client_and_runtime: tuple[TestClient, SessionRuntime]) -> None:
    client, runtime = client_and_runtime
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect(f"/ws/audio/{runtime.session.id}?token=wrong") as ws,
    ):
        ws.receive()


def test_full_spoken_interview_over_websocket(
    client_and_runtime: tuple[TestClient, SessionRuntime],
) -> None:
    client, runtime = client_and_runtime
    session = runtime.session
    url = f"/ws/audio/{session.id}?token={runtime.token}"
    with client.websocket_connect(url) as ws:
        snapshot = _receive(ws)
        assert isinstance(snapshot, dict) and snapshot["type"] == "snapshot"
        assert snapshot["state"] == "ready"

        ws.send_text(json.dumps({"type": "start"}))
        seen = finish_utterance(ws)
        audio = [item for item in seen if isinstance(item, bytes)]
        assert audio, "the first question must be streamed as audio"
        header, _ = decode_audio_chunk(audio[0])
        assert header.sample_rate > 0
        wait_state(ws, SessionState.LISTENING)

        for _ in range(2):
            ws.send_bytes(speech_pcm(1.0))
            ws.send_bytes(silence_pcm(1.5))
            turn, _ = receive_until(
                ws,
                lambda m: m["type"] == "transcript_turn" and m["turn"]["role"] == "candidate",
            )
            assert turn["turn"]["text"]
            finish_utterance(ws)
            wait_state(ws, SessionState.LISTENING)

        ws.send_text(json.dumps({"type": "end"}))
        completed, _ = receive_until(ws, lambda m: m["type"] == "completed")
        assert completed["report_ready"] is False

    assert session.state is SessionState.COMPLETED
    assert len(session.candidate_turns()) == 2
    assert session.question_count == 3


def test_malformed_messages_are_ignored_and_reconnect_resumes(
    client_and_runtime: tuple[TestClient, SessionRuntime],
) -> None:
    client, runtime = client_and_runtime
    url = f"/ws/audio/{runtime.session.id}?token={runtime.token}"
    with client.websocket_connect(url) as ws:
        _receive(ws)
        ws.send_text("not json")
        ws.send_text(json.dumps({"type": "unknown"}))
        ws.send_text(json.dumps({"type": "start"}))
        finish_utterance(ws)
        wait_state(ws, SessionState.LISTENING)

    assert runtime.orchestrator.state is SessionState.PAUSED

    with client.websocket_connect(url) as ws:
        snapshot = _receive(ws)
        assert isinstance(snapshot, dict) and snapshot["auto_paused"] is True
        wait_state(ws, SessionState.LISTENING)
        ws.send_text(json.dumps({"type": "text_answer", "text": "Typed answer after reconnect."}))
        receive_until(
            ws, lambda m: m["type"] == "transcript_turn" and m["turn"]["role"] == "candidate"
        )
