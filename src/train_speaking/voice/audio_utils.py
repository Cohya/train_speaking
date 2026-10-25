"""PCM conversion, resampling, framing, and WAV output."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from train_speaking.models.audio import FloatAudio

INT16_SCALE = 32768.0


def pcm16_to_float32(data: bytes) -> FloatAudio:
    if len(data) % 2:
        raise ValueError("int16 PCM payload must have an even number of bytes")
    return (np.frombuffer(data, dtype="<i2").astype(np.float32) / INT16_SCALE).astype(np.float32)


def float32_to_pcm16(audio: FloatAudio) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * (INT16_SCALE - 1)).astype("<i2").tobytes()


def resample_linear(audio: FloatAudio, src_rate: int, dst_rate: int) -> FloatAudio:
    """Linear-interpolation resampling; adequate for speech playback and VAD input."""
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be positive")
    if src_rate == dst_rate or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    n_out = round(len(audio) * dst_rate / src_rate)
    src_positions = np.linspace(0, len(audio) - 1, num=n_out)
    return np.interp(src_positions, np.arange(len(audio)), audio).astype(np.float32)


def rms(audio: FloatAudio) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


class FrameChunker:
    """Accumulates arbitrary-length PCM payloads and yields fixed-size float32 frames."""

    def __init__(self, frame_samples: int) -> None:
        if frame_samples <= 0:
            raise ValueError("frame_samples must be positive")
        self._frame_samples = frame_samples
        self._pending = np.zeros(0, dtype=np.float32)

    def push(self, pcm16: bytes) -> list[FloatAudio]:
        self._pending = np.concatenate([self._pending, pcm16_to_float32(pcm16)])
        n_frames = len(self._pending) // self._frame_samples
        frames = [
            self._pending[i * self._frame_samples : (i + 1) * self._frame_samples]
            for i in range(n_frames)
        ]
        self._pending = self._pending[n_frames * self._frame_samples :]
        return frames

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)


def write_wav(path: Path, audio: FloatAudio, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(float32_to_pcm16(audio))


def read_wav(path: Path) -> tuple[FloatAudio, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(f"{path.name}: expected mono 16-bit WAV")
        return pcm16_to_float32(wav.readframes(wav.getnframes())), wav.getframerate()
