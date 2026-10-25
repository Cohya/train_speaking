"""Voice activity detection on 512-sample, 16 kHz frames."""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol

from train_speaking.models.audio import FloatAudio
from train_speaking.providers.torch_runtime import import_torch
from train_speaking.voice.audio_utils import rms
from train_speaking.voice.protocol import INPUT_SAMPLE_RATE, VAD_FRAME_SAMPLES

logger = logging.getLogger(__name__)


class VoiceActivityDetector(Protocol):
    def speech_probability(self, frame: FloatAudio) -> float: ...

    def reset(self) -> None: ...


class EnergyVAD:
    """RMS-threshold VAD: dependency-free fallback, less robust to background noise."""

    def __init__(self, threshold_rms: float = 0.015, full_scale_rms: float = 0.05) -> None:
        if not 0 < threshold_rms < full_scale_rms:
            raise ValueError("expected 0 < threshold_rms < full_scale_rms")
        self._threshold = threshold_rms
        self._full = full_scale_rms

    def speech_probability(self, frame: FloatAudio) -> float:
        level = rms(frame)
        if level <= self._threshold:
            return 0.5 * level / self._threshold
        return min(1.0, 0.5 + 0.5 * (level - self._threshold) / (self._full - self._threshold))

    def reset(self) -> None:
        return None


class SileroVAD:
    """Silero VAD (torch). The model is stateful, so each session needs its own instance."""

    _load_lock = threading.Lock()

    def __init__(self) -> None:
        torch = import_torch()
        from silero_vad import load_silero_vad

        with self._load_lock:
            self._model: Any = load_silero_vad()
        self._torch: Any = torch

    def speech_probability(self, frame: FloatAudio) -> float:
        if frame.shape != (VAD_FRAME_SAMPLES,):
            raise ValueError(f"Silero VAD expects {VAD_FRAME_SAMPLES} samples, got {frame.shape}")
        with self._torch.no_grad():
            return float(self._model(self._torch.from_numpy(frame), INPUT_SAMPLE_RATE).item())

    def reset(self) -> None:
        self._model.reset_states()


def create_vad(provider: str) -> VoiceActivityDetector:
    if provider == "silero":
        try:
            return SileroVAD()
        except ImportError:
            logger.warning("silero-vad is not installed; falling back to energy VAD")
        except OSError as exc:
            logger.warning("torch failed to load; falling back to energy VAD: %s", exc)
    return EnergyVAD()
