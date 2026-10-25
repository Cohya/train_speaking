"""Silent text-to-speech for tests and text-only mode."""

from __future__ import annotations

import numpy as np

from train_speaking.models.audio import SynthesizedSpeech
from train_speaking.providers.tts.base import TTSError

FAKE_TTS_SAMPLE_RATE = 16_000
SECONDS_PER_WORD = 0.3


class FakeTTS:
    def __init__(self, fail_times: int = 0) -> None:
        self._fail_times = fail_times
        self.calls: list[str] = []

    async def warmup(self) -> None:
        return None

    async def synthesize(self, text: str) -> SynthesizedSpeech:
        self.calls.append(text)
        if self._fail_times > 0:
            self._fail_times -= 1
            raise TTSError("fake synthesis failure")
        words = text.split()
        n_samples = int(max(1, len(words)) * SECONDS_PER_WORD * FAKE_TTS_SAMPLE_RATE)
        timeline = [
            (i * SECONDS_PER_WORD, (i + 1) * SECONDS_PER_WORD, word) for i, word in enumerate(words)
        ]
        return SynthesizedSpeech(
            audio=np.zeros(n_samples, dtype=np.float32),
            sample_rate=FAKE_TTS_SAMPLE_RATE,
            phoneme_timeline=timeline,
        )


class NoTTS:
    """Always fails, so the orchestrator falls back to showing text only."""

    async def warmup(self) -> None:
        return None

    async def synthesize(self, text: str) -> SynthesizedSpeech:
        raise TTSError("text-to-speech is disabled")
