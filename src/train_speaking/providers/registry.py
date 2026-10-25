"""Builds providers from settings. Speech models are shared; VAD state is per session."""

from __future__ import annotations

from typing import Literal

from train_speaking.providers.llm.base import LLMProvider
from train_speaking.providers.stt.base import STTProvider
from train_speaking.providers.stt.fake import FakeSTT
from train_speaking.providers.tts.base import TTSProvider
from train_speaking.providers.tts.fake import FakeTTS, NoTTS
from train_speaking.settings import AppSettings
from train_speaking.voice.vad import VoiceActivityDetector, create_vad


def build_stt(settings: AppSettings) -> STTProvider:
    if settings.stt_provider == "fake":
        return FakeSTT(default="This is a placeholder answer from the fake transcriber.")
    from train_speaking.providers.stt.faster_whisper_stt import FasterWhisperSTT

    return FasterWhisperSTT(
        model=settings.stt_model,
        download_root=settings.models_dir / "whisper",
        device=settings.stt_device,
        cpu_fallback_model=settings.stt_cpu_fallback_model,
    )


def build_tts(settings: AppSettings) -> TTSProvider:
    if settings.tts_provider == "fake":
        return FakeTTS()
    if settings.tts_provider == "none":
        return NoTTS()
    from train_speaking.providers.tts.kokoro_tts import KokoroTTS

    return KokoroTTS(voice=settings.tts_voice, speed=settings.tts_speed)


def build_vad(settings: AppSettings) -> VoiceActivityDetector:
    return create_vad(settings.vad_provider)


def build_llm(
    settings: AppSettings, role: Literal["interviewer", "evaluator"] = "interviewer"
) -> LLMProvider | None:
    if settings.llm_provider == "none":
        return None
    from train_speaking.providers.llm.openai_compatible import OpenAICompatibleLLM

    if role == "evaluator":
        return OpenAICompatibleLLM(
            base_url=settings.eval_llm_base_url or settings.llm_base_url,
            model=settings.eval_llm_model or settings.llm_model,
            api_key=settings.eval_llm_api_key or settings.llm_api_key,
            timeout_s=settings.llm_timeout_s,
            is_cloud=settings.llm_is_cloud,
        )
    return OpenAICompatibleLLM(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        timeout_s=settings.llm_timeout_s,
        is_cloud=settings.llm_is_cloud,
    )
