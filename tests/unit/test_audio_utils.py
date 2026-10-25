from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from train_speaking.models.audio import AudioClip
from train_speaking.voice.audio_utils import (
    FrameChunker,
    float32_to_pcm16,
    pcm16_to_float32,
    read_wav,
    resample_linear,
    rms,
    write_wav,
)
from train_speaking.voice.protocol import (
    AudioChunkHeader,
    decode_audio_chunk,
    encode_audio_chunk,
)


def test_pcm_round_trip() -> None:
    audio = np.array([0.0, 0.5, -0.5, 0.999, -1.0], dtype=np.float32)
    restored = pcm16_to_float32(float32_to_pcm16(audio))
    assert restored.dtype == np.float32
    np.testing.assert_allclose(restored, audio, atol=1e-3)


def test_pcm_clips_out_of_range() -> None:
    restored = pcm16_to_float32(float32_to_pcm16(np.array([2.0, -3.0], dtype=np.float32)))
    assert restored.max() <= 1.0 and restored.min() >= -1.0


def test_odd_length_pcm_rejected() -> None:
    with pytest.raises(ValueError):
        pcm16_to_float32(b"\x00\x01\x02")


def test_resample_lengths_and_identity() -> None:
    audio = np.sin(np.linspace(0, 20, 48_000)).astype(np.float32)
    down = resample_linear(audio, 48_000, 16_000)
    assert len(down) == 16_000 and down.dtype == np.float32
    assert resample_linear(audio, 16_000, 16_000) is audio
    with pytest.raises(ValueError):
        resample_linear(audio, 0, 16_000)


def test_rms() -> None:
    assert rms(np.zeros(0, dtype=np.float32)) == 0.0
    assert rms(np.full(100, 0.5, dtype=np.float32)) == pytest.approx(0.5)


def test_frame_chunker_handles_partial_payloads() -> None:
    chunker = FrameChunker(4)
    samples = (np.arange(10) / 100).astype(np.float32)
    payload = float32_to_pcm16(samples)
    assert chunker.push(payload[:6]) == []
    frames = chunker.push(payload[6:])
    assert len(frames) == 2
    np.testing.assert_allclose(np.concatenate(frames), samples[:8], atol=1e-3)
    chunker.reset()
    assert chunker.push(float32_to_pcm16(samples[:3])) == []


def test_wav_round_trip(tmp_path: Path) -> None:
    audio = (np.sin(np.linspace(0, 100, 1600)) * 0.3).astype(np.float32)
    path = tmp_path / "nested" / "clip.wav"
    write_wav(path, audio, 16_000)
    restored, rate = read_wav(path)
    assert rate == 16_000
    np.testing.assert_allclose(restored, audio, atol=1e-3)


def test_audio_chunk_header_round_trip() -> None:
    header = AudioChunkHeader(utterance_id=7, seq=2, sample_rate=24_000)
    decoded, pcm = decode_audio_chunk(encode_audio_chunk(header, b"\x01\x02"))
    assert decoded == header and pcm == b"\x01\x02"
    with pytest.raises(ValueError):
        decode_audio_chunk(b"\x00")


def test_audio_clip_validation() -> None:
    with pytest.raises(ValueError):
        AudioClip(np.zeros((2, 2), dtype=np.float32), 16_000)
    with pytest.raises(ValueError):
        AudioClip(np.zeros(4, dtype=np.float64), 16_000)
    assert AudioClip(np.zeros(8000, dtype=np.float32), 16_000).duration_s == 0.5
