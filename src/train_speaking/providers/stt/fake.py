"""Deterministic speech-to-text for tests and offline development."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from train_speaking.models.audio import AudioClip
from train_speaking.models.transcript import Transcription, WordTiming
from train_speaking.providers.stt.base import STTError

SECONDS_PER_WORD = 0.4


class FakeSTT:
    """Returns scripted transcripts in order; an `Exception` entry is raised instead."""

    def __init__(self, scripted: Iterable[str | Exception] = (), default: str = "") -> None:
        self._queue: deque[str | Exception] = deque(scripted)
        self._default = default
        self.calls: list[AudioClip] = []

    def add(self, item: str | Exception) -> None:
        self._queue.append(item)

    async def warmup(self) -> None:
        return None

    async def transcribe(
        self, clip: AudioClip, *, language: str | None = None, prompt: str | None = None
    ) -> Transcription:
        self.calls.append(clip)
        item = self._queue.popleft() if self._queue else self._default
        if isinstance(item, Exception):
            if isinstance(item, STTError):
                raise item
            raise STTError(str(item)) from item
        words = [
            WordTiming(
                word=w,
                start=i * SECONDS_PER_WORD,
                end=(i + 0.8) * SECONDS_PER_WORD,
                probability=0.95,
            )
            for i, w in enumerate(item.split())
        ]
        return Transcription(text=item, words=words, language=language, duration_s=clip.duration_s)
