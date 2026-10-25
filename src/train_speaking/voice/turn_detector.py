"""Turn-end detection from per-frame speech probabilities.

Pure logic with no audio-model dependency, so behavior is unit-testable with synthetic
probability sequences. Callers pass the session-elapsed time of each frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from train_speaking.models.audio import AudioClip, FloatAudio
from train_speaking.voice.protocol import FRAME_MS, INPUT_SAMPLE_RATE


@dataclass(frozen=True)
class TurnDetectorConfig:
    speech_threshold: float = 0.5
    # Hysteresis: once speaking, a frame only counts as silence below this probability.
    silence_threshold: float = 0.35
    speech_start_ms: float = 96.0
    end_silence_ms: float = 1200.0
    min_speech_ms: float = 400.0
    pre_roll_ms: float = 300.0
    max_answer_s: float = 180.0
    barge_in_threshold: float = 0.7
    barge_in_min_ms: float = 400.0

    def __post_init__(self) -> None:
        if not 0 < self.silence_threshold <= self.speech_threshold <= 1:
            raise ValueError("expected 0 < silence_threshold <= speech_threshold <= 1")


class ListenMode(StrEnum):
    IGNORE = "ignore"
    LISTEN = "listen"
    BARGE_IN = "barge_in"


class TurnEventType(StrEnum):
    SPEECH_STARTED = "speech_started"
    TURN_ENDED = "turn_ended"
    SPEECH_DISCARDED = "speech_discarded"
    MAX_ANSWER_REACHED = "max_answer_reached"
    BARGE_IN = "barge_in"


@dataclass(frozen=True)
class TurnEvent:
    type: TurnEventType
    at: float
    clip: AudioClip | None = None


class TurnDetector:
    def __init__(self, config: TurnDetectorConfig | None = None) -> None:
        self.config = config or TurnDetectorConfig()
        self._mode = ListenMode.IGNORE
        self._pre_roll: deque[tuple[float, FloatAudio]] = deque(
            maxlen=max(1, round(self.config.pre_roll_ms / FRAME_MS))
        )
        self._reset_utterance()

    @property
    def mode(self) -> ListenMode:
        return self._mode

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def set_mode(self, mode: ListenMode) -> None:
        if mode is not self._mode:
            self._mode = mode
            self._pre_roll.clear()
            self._reset_utterance()

    def _reset_utterance(self) -> None:
        self._in_speech = False
        self._onset: list[tuple[float, FloatAudio]] = []
        self._buffer: list[tuple[float, FloatAudio]] = []
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._speech_started_at: float | None = None

    def current_audio(self) -> AudioClip | None:
        """Audio buffered for the in-progress answer (used for interim transcripts)."""
        return self._clip() if self._in_speech else None

    def feed(self, frame: FloatAudio, probability: float, now: float) -> TurnEvent | None:
        if self._mode is ListenMode.IGNORE:
            return None
        if self._mode is ListenMode.BARGE_IN:
            return self._feed_barge_in(frame, probability, now)
        if not self._in_speech:
            return self._feed_waiting(frame, probability, now)
        return self._feed_speaking(frame, probability, now)

    def force_end(self, now: float) -> TurnEvent | None:
        """End the turn immediately (the user pressed "I'm done")."""
        if not self._in_speech:
            self._reset_utterance()
            return None
        return self._finish(now, TurnEventType.TURN_ENDED)

    def _feed_waiting(self, frame: FloatAudio, probability: float, now: float) -> TurnEvent | None:
        if probability >= self.config.speech_threshold:
            self._onset.append((now, frame))
            if len(self._onset) * FRAME_MS >= self.config.speech_start_ms:
                return self._start_speech(now)
            return None
        self._pre_roll.extend(self._onset)
        self._onset = []
        self._pre_roll.append((now, frame))
        return None

    def _start_speech(self, now: float) -> TurnEvent:
        self._in_speech = True
        self._speech_started_at = self._onset[0][0]
        self._buffer = [*self._pre_roll, *self._onset]
        self._speech_ms = len(self._onset) * FRAME_MS
        self._onset = []
        self._pre_roll.clear()
        return TurnEvent(TurnEventType.SPEECH_STARTED, now)

    def _feed_speaking(self, frame: FloatAudio, probability: float, now: float) -> TurnEvent | None:
        self._buffer.append((now, frame))
        if probability >= self.config.silence_threshold:
            self._speech_ms += FRAME_MS
            self._silence_ms = 0.0
        else:
            self._silence_ms += FRAME_MS
        if len(self._buffer) * FRAME_MS >= self.config.max_answer_s * 1000:
            return self._finish(now, TurnEventType.MAX_ANSWER_REACHED)
        if self._silence_ms >= self.config.end_silence_ms:
            if self._speech_ms >= self.config.min_speech_ms:
                return self._finish(now, TurnEventType.TURN_ENDED)
            self._reset_utterance()
            return TurnEvent(TurnEventType.SPEECH_DISCARDED, now)
        return None

    def _feed_barge_in(self, frame: FloatAudio, probability: float, now: float) -> TurnEvent | None:
        if probability >= self.config.barge_in_threshold:
            self._onset.append((now, frame))
            if len(self._onset) * FRAME_MS >= self.config.barge_in_min_ms:
                self._mode = ListenMode.LISTEN
                self._start_speech(now)
                return TurnEvent(TurnEventType.BARGE_IN, now)
        else:
            self._onset = []
        return None

    def _finish(self, now: float, event_type: TurnEventType) -> TurnEvent:
        clip = self._clip()
        self._reset_utterance()
        return TurnEvent(event_type, now, clip)

    def _clip(self) -> AudioClip:
        samples: FloatAudio = np.concatenate([f for _, f in self._buffer]).astype(np.float32)
        return AudioClip(
            samples=samples,
            sample_rate=INPUT_SAMPLE_RATE,
            started_at=self._buffer[0][0],
            speech_started_at=self._speech_started_at,
        )
