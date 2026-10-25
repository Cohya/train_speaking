"""Local text-to-speech with Kokoro-82M."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import numpy as np

from train_speaking.models.audio import SynthesizedSpeech
from train_speaking.providers.torch_runtime import import_torch
from train_speaking.providers.tts.base import TTSError

logger = logging.getLogger(__name__)

KOKORO_SAMPLE_RATE = 24_000
KOKORO_REPO = "hexgrad/Kokoro-82M"


class KokoroTTS:
    def __init__(self, voice: str = "af_heart", speed: float = 1.0, lang_code: str = "a") -> None:
        self._voice = voice
        self._speed = speed
        self._lang_code = lang_code
        self._pipeline: Any = None
        # The Kokoro pipeline is not thread-safe.
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._pipeline is None:
            torch = import_torch()
            from kokoro import KPipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipeline = KPipeline(lang_code=self._lang_code, repo_id=KOKORO_REPO, device=device)
            # Fetch the voice now so download failures surface here, not mid-sentence.
            pipeline.load_voice(self._voice)
            self._pipeline = pipeline
            logger.info("kokoro loaded", extra={"device": device, "voice": self._voice})
        return self._pipeline

    async def warmup(self) -> None:
        await asyncio.to_thread(self._synthesize_sync, "Hello.")

    async def synthesize(self, text: str) -> SynthesizedSpeech:
        return await asyncio.to_thread(self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> SynthesizedSpeech:
        with self._lock:
            try:
                pipeline = self._load()
            except Exception as exc:
                # Loading can fail in many library-specific ways (missing weights, network
                # errors during the first download, CUDA issues); all mean "no TTS".
                raise TTSError(f"kokoro could not be loaded: {exc}") from exc
            try:
                chunks: list[np.ndarray[Any, np.dtype[np.float32]]] = []
                timeline: list[tuple[float, float, str]] = []
                offset = 0.0
                for result in pipeline(text, voice=self._voice, speed=self._speed):
                    if result.audio is None:
                        continue
                    audio = result.audio.detach().cpu().numpy().astype(np.float32)
                    for token in result.tokens or []:
                        if token.start_ts is not None and token.end_ts is not None:
                            timeline.append(
                                (
                                    offset + token.start_ts,
                                    offset + token.end_ts,
                                    token.phonemes or "",
                                )
                            )
                    chunks.append(audio)
                    offset += len(audio) / KOKORO_SAMPLE_RATE
            except (RuntimeError, ValueError, OSError) as exc:
                raise TTSError(f"kokoro synthesis failed: {exc}") from exc
        if not chunks:
            raise TTSError("kokoro produced no audio")
        return SynthesizedSpeech(
            audio=np.concatenate(chunks), sample_rate=KOKORO_SAMPLE_RATE, phoneme_timeline=timeline
        )
