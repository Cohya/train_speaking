"""Speech-to-text provider interface."""

from __future__ import annotations

from typing import Protocol

from train_speaking.models.audio import AudioClip
from train_speaking.models.transcript import Transcription


class STTError(RuntimeError):
    pass


class STTProvider(Protocol):
    async def transcribe(
        self, clip: AudioClip, *, language: str | None = None, prompt: str | None = None
    ) -> Transcription:
        """Word timestamps in the result are relative to the start of `clip`."""
        ...

    async def warmup(self) -> None: ...
