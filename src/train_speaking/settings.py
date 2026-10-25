"""Application settings loaded from environment variables and an optional `.env` file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AppSettings(BaseSettings):
    """All settings use the `TS_` prefix, e.g. `TS_LLM_MODEL=qwen2.5:14b`."""

    model_config = SettingsConfigDict(
        env_prefix="TS_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    open_browser: bool = True

    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Transcript text is personal data; it is kept out of logs unless explicitly enabled.
    log_transcripts: bool = False

    stt_provider: Literal["faster_whisper", "fake"] = "faster_whisper"
    stt_model: str = "large-v3-turbo"
    stt_cpu_fallback_model: str = "small.en"
    stt_device: Literal["auto", "cuda", "cpu"] = "auto"

    tts_provider: Literal["kokoro", "fake", "none"] = "kokoro"
    tts_voice: str = "af_heart"
    tts_speed: float = Field(default=1.0, ge=0.5, le=2.0)

    vad_provider: Literal["silero", "energy"] = "silero"

    llm_provider: Literal["openai_compatible", "none"] = "none"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:14b"
    llm_api_key: SecretStr | None = None
    llm_timeout_s: float = Field(default=30.0, gt=0)
    llm_is_cloud: bool = False

    # The evaluator may use a different (usually stronger) model. Unset values reuse the
    # interviewer LLM settings.
    eval_llm_base_url: str | None = None
    eval_llm_model: str | None = None
    eval_llm_api_key: SecretStr | None = None

    max_tokens_per_call: int = Field(default=800, ge=64)
    session_token_budget: int = Field(default=80_000, ge=1_000)
    cost_per_1k_input_tokens: float = Field(default=0.0, ge=0)
    cost_per_1k_output_tokens: float = Field(default=0.0, ge=0)

    interim_transcripts: bool = False
    avatar_mode: Literal["amplitude", "viseme"] = "amplitude"

    @property
    def tmp_sessions_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def saved_sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def uses_cloud(self) -> bool:
        return self.llm_provider == "openai_compatible" and self.llm_is_cloud
