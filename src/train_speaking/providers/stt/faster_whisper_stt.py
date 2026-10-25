"""Local speech-to-text with faster-whisper (CTranslate2)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Literal

from train_speaking.models.audio import AudioClip
from train_speaking.models.transcript import Transcription, WordTiming
from train_speaking.providers.stt.base import STTError
from train_speaking.providers.torch_runtime import import_torch
from train_speaking.voice.audio_utils import resample_linear
from train_speaking.voice.protocol import INPUT_SAMPLE_RATE

logger = logging.getLogger(__name__)

# Whisper tends to drop disfluencies; an initial prompt containing them makes it more
# likely to transcribe fillers verbatim. Filler counts remain a lower bound.
FILLER_BIASED_PROMPT = "Umm, let me think. Uh, so, I mean, we, um, we built the, uh, the service."


def register_cuda_dlls() -> None:
    """Make CUDA/cuDNN DLLs from pip wheels discoverable by CTranslate2 on Windows.

    Importing torch adds its bundled CUDA libraries to the DLL search path; `nvidia-*`
    wheels, if installed, are registered as well.
    """
    if sys.platform != "win32":
        return
    with contextlib.suppress(ImportError, OSError):
        import_torch()
    for site in map(Path, sys.path):
        nvidia = site / "nvidia"
        if nvidia.is_dir():
            for bin_dir in nvidia.glob("*/bin"):
                os.add_dll_directory(str(bin_dir))


class FasterWhisperSTT:
    def __init__(
        self,
        model: str,
        download_root: Path,
        device: Literal["auto", "cuda", "cpu"] = "auto",
        cpu_fallback_model: str = "small.en",
    ) -> None:
        self._model_name = model
        self._download_root = download_root
        self._device = device
        self._cpu_fallback_model = cpu_fallback_model
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            register_cuda_dlls()
            from faster_whisper import WhisperModel

            root = str(self._download_root)
            if self._device in ("auto", "cuda"):
                try:
                    self._model = WhisperModel(
                        self._model_name, device="cuda", compute_type="float16", download_root=root
                    )
                    logger.info(
                        "whisper loaded", extra={"model": self._model_name, "device": "cuda"}
                    )
                    return self._model
                # Failures range from missing CUDA DLLs to network errors while downloading
                # weights; any of them means falling back (or failing) the same way.
                except Exception as exc:
                    if self._device == "cuda":
                        raise STTError(f"could not load whisper on CUDA: {exc}") from exc
                    logger.warning("whisper CUDA load failed; using CPU", extra={"error": str(exc)})
            try:
                self._model = WhisperModel(
                    self._cpu_fallback_model, device="cpu", compute_type="int8", download_root=root
                )
            except Exception as exc:
                raise STTError(f"could not load whisper: {exc}") from exc
            logger.info(
                "whisper loaded", extra={"model": self._cpu_fallback_model, "device": "cpu"}
            )
            return self._model

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load)

    async def transcribe(
        self, clip: AudioClip, *, language: str | None = None, prompt: str | None = None
    ) -> Transcription:
        return await asyncio.to_thread(self._transcribe_sync, clip, language, prompt)

    def _transcribe_sync(
        self, clip: AudioClip, language: str | None, prompt: str | None
    ) -> Transcription:
        model = self._load()
        audio = resample_linear(clip.samples, clip.sample_rate, INPUT_SAMPLE_RATE)
        try:
            segments, info = model.transcribe(
                audio,
                language=language,
                initial_prompt=prompt or FILLER_BIASED_PROMPT,
                word_timestamps=True,
                beam_size=5,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            segment_list = list(segments)
        except (RuntimeError, ValueError) as exc:
            raise STTError(f"transcription failed: {exc}") from exc
        words = [
            WordTiming(
                word=w.word.strip(),
                start=float(w.start),
                end=float(w.end),
                probability=float(w.probability),
            )
            for segment in segment_list
            for w in (segment.words or [])
            if w.word.strip()
        ]
        text = " ".join(segment.text.strip() for segment in segment_list).strip()
        return Transcription(
            text=text, words=words, language=info.language, duration_s=clip.duration_s
        )
