"""Shared test harness: fake transport, synthetic speech, and orchestrator factory."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import numpy as np

from train_speaking.interviewer.base import Interviewer
from train_speaking.interviewer.scripted import ScriptedInterviewer
from train_speaking.models.config import InterviewConfig
from train_speaking.models.session import Session, SessionState
from train_speaking.providers.stt.fake import FakeSTT
from train_speaking.providers.tts.base import TTSProvider
from train_speaking.providers.tts.fake import FakeTTS
from train_speaking.session.clock import FakeClock
from train_speaking.session.orchestrator import (
    AnswerRecorder,
    Evaluator,
    InterviewOrchestrator,
    OrchestratorOptions,
)
from train_speaking.voice.audio_utils import float32_to_pcm16
from train_speaking.voice.listener import AnswerListener
from train_speaking.voice.protocol import INPUT_SAMPLE_RATE
from train_speaking.voice.turn_detector import TurnDetector, TurnDetectorConfig
from train_speaking.voice.vad import EnergyVAD

TEST_OPTIONS = OrchestratorOptions(
    half_duplex_tail_s=0.0, run_ticker=False, playback_timeout_margin_s=2.0
)


def speech_pcm(seconds: float, amplitude: float = 0.3) -> bytes:
    t = np.arange(int(seconds * INPUT_SAMPLE_RATE)) / INPUT_SAMPLE_RATE
    return float32_to_pcm16((amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32))


def silence_pcm(seconds: float) -> bytes:
    return bytes(int(seconds * INPUT_SAMPLE_RATE) * 2)


class RecordingTransport:
    """Collects messages; confirms playback automatically unless `auto_ack` is False."""

    def __init__(self, orchestrator: InterviewOrchestrator, auto_ack: bool = True) -> None:
        self.orchestrator = orchestrator
        self.auto_ack = auto_ack
        self.messages: list[dict[str, Any]] = []
        self.audio: list[bytes] = []

    async def send_json(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        if message["type"] == "utterance_end" and self.auto_ack:
            asyncio.get_running_loop().call_soon(
                self.orchestrator.playback_finished, message["utterance_id"]
            )

    async def send_bytes(self, data: bytes) -> None:
        self.audio.append(data)

    def of_type(self, message_type: str) -> list[dict[str, Any]]:
        return [m for m in self.messages if m["type"] == message_type]


async def wait_for(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not met before timeout")
        await asyncio.sleep(0.005)


async def wait_for_state(
    orchestrator: InterviewOrchestrator, state: SessionState, timeout: float = 3.0
) -> None:
    try:
        await wait_for(lambda: orchestrator.state is state, timeout)
    except AssertionError as exc:
        raise AssertionError(f"expected {state}, still {orchestrator.state}") from exc


def make_orchestrator(
    config: InterviewConfig,
    *,
    stt: FakeSTT | None = None,
    tts: TTSProvider | None = None,
    interviewer: Interviewer | None = None,
    evaluator: Evaluator | None = None,
    recorder: AnswerRecorder | None = None,
    clock: FakeClock | None = None,
    options: OrchestratorOptions = TEST_OPTIONS,
    session_id: str = "test-session-0001",
) -> InterviewOrchestrator:
    session = Session(id=session_id, config=config, state=SessionState.READY)
    detector = TurnDetector(TurnDetectorConfig(end_silence_ms=float(config.end_silence_ms)))
    return InterviewOrchestrator(
        session,
        interviewer=interviewer or ScriptedInterviewer(),
        stt=stt or FakeSTT(default="A default answer."),
        tts=tts or FakeTTS(),
        listener=AnswerListener(EnergyVAD(), detector),
        clock=clock or FakeClock(),
        evaluator=evaluator,
        recorder=recorder,
        options=options,
    )


async def answer_by_voice(
    orchestrator: InterviewOrchestrator, speech_s: float = 1.0, silence_s: float = 1.5
) -> None:
    await orchestrator.on_audio(speech_pcm(speech_s))
    await orchestrator.on_audio(silence_pcm(silence_s))
