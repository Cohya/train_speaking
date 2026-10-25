from __future__ import annotations

import numpy as np
import pytest

from train_speaking.voice.protocol import FRAME_MS, VAD_FRAME_SAMPLES
from train_speaking.voice.turn_detector import (
    ListenMode,
    TurnDetector,
    TurnDetectorConfig,
    TurnEvent,
    TurnEventType,
)

FRAME_S = FRAME_MS / 1000


def frames_for(ms: float) -> int:
    return round(ms / FRAME_MS)


class Driver:
    """Feeds synthetic probability sequences with a synthetic clock."""

    def __init__(self, detector: TurnDetector) -> None:
        self.detector = detector
        self.now = 0.0
        self.events: list[TurnEvent] = []

    def feed(self, probability: float, ms: float) -> None:
        for _ in range(frames_for(ms)):
            frame = np.full(VAD_FRAME_SAMPLES, probability, dtype=np.float32)
            event = self.detector.feed(frame, probability, self.now)
            if event is not None:
                self.events.append(event)
            self.now += FRAME_S

    def types(self) -> list[TurnEventType]:
        return [e.type for e in self.events]


@pytest.fixture
def driver() -> Driver:
    detector = TurnDetector(TurnDetectorConfig(end_silence_ms=1200, min_speech_ms=400))
    detector.set_mode(ListenMode.LISTEN)
    return Driver(detector)


def test_ignore_mode_produces_no_events() -> None:
    driver = Driver(TurnDetector())
    driver.feed(0.9, 2000)
    assert driver.events == []


def test_speech_then_silence_ends_turn(driver: Driver) -> None:
    driver.feed(0.0, 500)
    driver.feed(0.9, 1500)
    driver.feed(0.0, 1300)
    assert driver.types() == [TurnEventType.SPEECH_STARTED, TurnEventType.TURN_ENDED]
    clip = driver.events[-1].clip
    assert clip is not None
    # Pre-roll (300 ms) + speech (1500 ms) + trailing silence up to the end threshold.
    assert clip.duration_s == pytest.approx(0.3 + 1.5 + 1.2, abs=0.1)
    assert clip.speech_started_at == pytest.approx(0.5, abs=FRAME_S)
    assert clip.started_at == pytest.approx(0.2, abs=FRAME_S)


def test_thinking_pause_shorter_than_threshold_does_not_end_turn(driver: Driver) -> None:
    driver.feed(0.9, 1000)
    driver.feed(0.1, 1000)
    driver.feed(0.9, 1000)
    driver.feed(0.0, 1300)
    assert driver.types() == [TurnEventType.SPEECH_STARTED, TurnEventType.TURN_ENDED]
    clip = driver.events[-1].clip
    assert clip is not None and clip.duration_s > 3.0


def test_short_noise_burst_is_ignored(driver: Driver) -> None:
    driver.feed(0.9, 64)
    driver.feed(0.0, 2000)
    assert driver.events == []


def test_speech_shorter_than_minimum_is_discarded(driver: Driver) -> None:
    driver.feed(0.9, 200)
    driver.feed(0.0, 1300)
    assert driver.types() == [TurnEventType.SPEECH_STARTED, TurnEventType.SPEECH_DISCARDED]
    assert not driver.detector.in_speech


def test_hysteresis_keeps_quiet_speech_in_turn(driver: Driver) -> None:
    driver.feed(0.9, 500)
    driver.feed(0.4, 2000)
    assert driver.types() == [TurnEventType.SPEECH_STARTED]


def test_max_answer_length_forces_turn_end() -> None:
    detector = TurnDetector(TurnDetectorConfig(max_answer_s=3.0))
    detector.set_mode(ListenMode.LISTEN)
    driver = Driver(detector)
    driver.feed(0.9, 4000)
    assert driver.types()[:2] == [TurnEventType.SPEECH_STARTED, TurnEventType.MAX_ANSWER_REACHED]
    clip = driver.events[1].clip
    assert clip is not None and clip.duration_s == pytest.approx(3.0, abs=0.05)


def test_force_end_returns_clip_only_while_speaking(driver: Driver) -> None:
    assert driver.detector.force_end(driver.now) is None
    driver.feed(0.9, 600)
    event = driver.detector.force_end(driver.now)
    assert event is not None and event.type is TurnEventType.TURN_ENDED
    assert event.clip is not None
    assert not driver.detector.in_speech


def test_current_audio_reflects_in_progress_answer(driver: Driver) -> None:
    assert driver.detector.current_audio() is None
    driver.feed(0.9, 1000)
    clip = driver.detector.current_audio()
    assert clip is not None and clip.duration_s > 0.9


def test_barge_in_requires_sustained_loud_speech() -> None:
    detector = TurnDetector(TurnDetectorConfig(barge_in_min_ms=400, barge_in_threshold=0.7))
    detector.set_mode(ListenMode.BARGE_IN)
    driver = Driver(detector)
    driver.feed(0.6, 1000)
    driver.feed(0.9, 200)
    driver.feed(0.0, 100)
    assert driver.events == []
    driver.feed(0.9, 500)
    assert driver.types() == [TurnEventType.BARGE_IN]
    assert detector.mode is ListenMode.LISTEN and detector.in_speech
    driver.feed(0.0, 1300)
    assert driver.types()[-1] is TurnEventType.TURN_ENDED


def test_mode_change_resets_partial_utterance(driver: Driver) -> None:
    driver.feed(0.9, 600)
    driver.detector.set_mode(ListenMode.IGNORE)
    driver.detector.set_mode(ListenMode.LISTEN)
    assert not driver.detector.in_speech


def test_invalid_thresholds_rejected() -> None:
    with pytest.raises(ValueError):
        TurnDetectorConfig(speech_threshold=0.3, silence_threshold=0.5)
