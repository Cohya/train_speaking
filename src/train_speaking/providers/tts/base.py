"""Text-to-speech provider interface.

The orchestrator splits interviewer text into sentences and synthesizes them one at a time,
so playback of the first sentence can start while later sentences are still synthesizing.
"""

from __future__ import annotations

from typing import Protocol

from train_speaking.models.audio import SynthesizedSpeech


class TTSError(RuntimeError):
    pass


class TTSProvider(Protocol):
    async def synthesize(self, text: str) -> SynthesizedSpeech: ...

    async def warmup(self) -> None: ...
