"""`/ws/audio/{session_id}`: microphone audio in, interviewer audio and events out."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from train_speaking.models.events import (
    ClientErrorMessage,
    DoneAnsweringMessage,
    EndMessage,
    PauseMessage,
    PlaybackFinishedMessage,
    PlaybackStartedMessage,
    ResumeMessage,
    StartMessage,
    TextAnswerMessage,
    client_message_adapter,
)
from train_speaking.session.manager import SessionManager
from train_speaking.session.orchestrator import InterviewOrchestrator
from train_speaking.voice.protocol import MAX_CLIENT_AUDIO_MESSAGE_BYTES

logger = logging.getLogger(__name__)

POLICY_VIOLATION = 1008


class WebSocketTransport:
    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket

    async def send_json(self, message: dict[str, Any]) -> None:
        await self._websocket.send_json(message)

    async def send_bytes(self, data: bytes) -> None:
        await self._websocket.send_bytes(data)


async def dispatch_text(orchestrator: InterviewOrchestrator, raw: str) -> None:
    try:
        message = client_message_adapter.validate_json(raw)
    except ValidationError:
        logger.info("ignoring malformed client message")
        return
    match message:
        case StartMessage():
            await orchestrator.start()
        case PauseMessage():
            await orchestrator.pause()
        case ResumeMessage():
            await orchestrator.resume()
        case EndMessage():
            await orchestrator.end()
        case DoneAnsweringMessage():
            await orchestrator.done_answering()
        case TextAnswerMessage(text=text):
            await orchestrator.text_answer(text)
        case PlaybackStartedMessage(utterance_id=utterance_id):
            orchestrator.playback_started(utterance_id)
        case PlaybackFinishedMessage(utterance_id=utterance_id):
            orchestrator.playback_finished(utterance_id)
        case ClientErrorMessage(kind=kind, message=text):
            orchestrator.client_error(kind, text)


def create_ws_router(manager: SessionManager) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws/audio/{session_id}")
    async def audio_socket(websocket: WebSocket, session_id: str, token: str = "") -> None:
        runtime = manager.verify(session_id, token)
        if runtime is None:
            await websocket.close(code=POLICY_VIOLATION)
            return
        await websocket.accept()
        orchestrator = runtime.orchestrator
        transport = WebSocketTransport(websocket)
        await orchestrator.attach(transport)
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data:
                    if len(data) <= MAX_CLIENT_AUDIO_MESSAGE_BYTES:
                        await orchestrator.on_audio(data)
                    continue
                text = message.get("text")
                if text:
                    await dispatch_text(orchestrator, text)
        except WebSocketDisconnect:
            pass
        finally:
            await orchestrator.detach(transport)

    return router
