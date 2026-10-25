"""Loading job descriptions and building validated interview configurations."""

from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any

from pydantic import ValidationError

from train_speaking.models.config import InterviewConfig

MAX_UPLOAD_BYTES = 1_000_000
ALLOWED_EXTENSIONS = frozenset({".txt", ".md", ".pdf"})


class ConfigError(ValueError):
    """Raised with a user-presentable message when configuration input is invalid."""

    def __init__(self, messages: list[str]) -> None:
        super().__init__("; ".join(messages))
        self.messages = messages


def load_job_description(filename: str, content: bytes) -> str:
    extension = PurePath(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ConfigError([f"Unsupported file type '{extension or filename}'. Use {allowed}."])
    if len(content) > MAX_UPLOAD_BYTES:
        raise ConfigError([f"File is larger than {MAX_UPLOAD_BYTES // 1000} KB."])
    if extension == ".pdf":
        text = _pdf_to_text(content)
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ConfigError(["File is not valid UTF-8 text."]) from exc
    text = text.strip()
    if not text:
        raise ConfigError(["The uploaded file contains no text."])
    return text


def _pdf_to_text(content: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except PdfReadError as exc:
        raise ConfigError(["The PDF could not be read."]) from exc


def build_config(form: Mapping[str, Any]) -> InterviewConfig:
    try:
        return InterviewConfig.model_validate(dict(form))
    except ValidationError as exc:
        raise ConfigError([_format_error(err) for err in exc.errors()]) from exc


def _format_error(error: Mapping[str, Any]) -> str:
    location = ".".join(str(part) for part in error.get("loc", ()) if part != "__root__")
    message = str(error.get("msg", "invalid value")).removeprefix("Value error, ")
    return f"{location}: {message}" if location else message
