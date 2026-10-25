"""Real model smoke test: Kokoro speaks, Silero hears speech, faster-whisper transcribes it.

Opt-in (downloads model weights into ./models on first run):
    pytest -m "gpu and real_providers" tests/integration/test_real_speech.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
# Must be set before any Hugging Face library is imported.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "huggingface"))

from train_speaking.models.audio import AudioClip  # noqa: E402
from train_speaking.providers.stt.faster_whisper_stt import FasterWhisperSTT  # noqa: E402
from train_speaking.providers.tts.kokoro_tts import KokoroTTS  # noqa: E402
from train_speaking.voice.audio_utils import resample_linear  # noqa: E402
from train_speaking.voice.protocol import INPUT_SAMPLE_RATE, VAD_FRAME_SAMPLES  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.real_providers]

SENTENCE = "I led the migration of our payment service to a new database last year."
MIN_WORD_OVERLAP = 0.8
SPEECH_FRAME_RATIO = 0.3


async def test_tts_vad_stt_round_trip() -> None:
    from train_speaking.voice.vad import SileroVAD

    speech = await KokoroTTS().synthesize(SENTENCE)
    assert speech.duration_s > 1.0
    assert speech.phoneme_timeline, "Kokoro should report token timings for lip-sync"

    audio = resample_linear(speech.audio, speech.sample_rate, INPUT_SAMPLE_RATE)
    vad = SileroVAD()
    frames = len(audio) // VAD_FRAME_SAMPLES
    probabilities = [
        vad.speech_probability(audio[i * VAD_FRAME_SAMPLES : (i + 1) * VAD_FRAME_SAMPLES])
        for i in range(frames)
    ]
    assert np.mean(np.array(probabilities) > 0.5) > SPEECH_FRAME_RATIO

    stt = FasterWhisperSTT("large-v3-turbo", MODELS_DIR / "whisper")
    result = await stt.transcribe(AudioClip(audio, INPUT_SAMPLE_RATE), language="en")
    expected = {w.strip(".,").lower() for w in SENTENCE.split()}
    heard = {w.word.strip(" .,").lower() for w in result.words}
    assert len(expected & heard) / len(expected) >= MIN_WORD_OVERLAP, result.text
    assert all(w.end >= w.start for w in result.words)
