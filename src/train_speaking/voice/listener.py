"""Combines framing, VAD, and turn detection for one session's microphone stream."""

from __future__ import annotations

from train_speaking.models.audio import AudioClip
from train_speaking.voice.audio_utils import FrameChunker
from train_speaking.voice.protocol import FRAME_MS, VAD_FRAME_SAMPLES
from train_speaking.voice.turn_detector import ListenMode, TurnDetector, TurnEvent
from train_speaking.voice.vad import VoiceActivityDetector


class AnswerListener:
    def __init__(self, vad: VoiceActivityDetector, detector: TurnDetector) -> None:
        self._vad = vad
        self._detector = detector
        self._chunker = FrameChunker(VAD_FRAME_SAMPLES)
        self.last_probability = 0.0

    @property
    def mode(self) -> ListenMode:
        return self._detector.mode

    @property
    def in_speech(self) -> bool:
        return self._detector.in_speech

    def set_mode(self, mode: ListenMode) -> None:
        if mode is not self._detector.mode:
            self._vad.reset()
            self._chunker.reset()
        self._detector.set_mode(mode)

    def feed_pcm(self, pcm16: bytes, now: float) -> list[TurnEvent]:
        """Process one network payload; `now` is the session time of its last sample."""
        frames = self._chunker.push(pcm16)
        events: list[TurnEvent] = []
        for index, frame in enumerate(frames):
            frame_time = now - (len(frames) - 1 - index) * FRAME_MS / 1000
            if self._detector.mode is ListenMode.IGNORE:
                continue
            self.last_probability = self._vad.speech_probability(frame)
            event = self._detector.feed(frame, self.last_probability, frame_time)
            if event is not None:
                events.append(event)
        return events

    def force_end(self, now: float) -> TurnEvent | None:
        return self._detector.force_end(now)

    def current_audio(self) -> AudioClip | None:
        return self._detector.current_audio()
