from __future__ import annotations

from typing import Any

import pytest

from tests.conftest import make_config
from train_speaking.config.loader import (
    MAX_UPLOAD_BYTES,
    ConfigError,
    build_config,
    load_job_description,
)
from train_speaking.models.config import ConsentFlags, EvaluationCategory


def _form(**overrides: Any) -> dict[str, Any]:
    form: dict[str, Any] = {
        "job_description": "Backend engineer working on Python services and APIs.",
        "rules": "  Ask one question at a time.  ",
        "duration_min": 8,
        "consent": {"microphone": True},
    }
    form.update(overrides)
    return form


def test_build_config_strips_text_and_uses_defaults() -> None:
    config = build_config(_form())
    assert config.rules == "Ask one question at a time."
    assert config.duration_s == 480
    assert EvaluationCategory.PACE in config.criteria
    assert config.consent.record_audio is False


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"job_description": "too short"}, "job_description"),
        ({"job_description": " " * 30 + "x" * 5}, "job_description"),
        ({"duration_min": 2}, "duration_min"),
        ({"duration_min": 31}, "duration_min"),
        ({"rules": "x" * 4001}, "rules"),
        ({"criteria": []}, "at least one"),
        ({"consent": {"microphone": False}}, "microphone consent"),
        ({"end_silence_ms": 100}, "end_silence_ms"),
    ],
)
def test_build_config_rejects_invalid_input(overrides: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ConfigError) as info:
        build_config(_form(**overrides))
    assert fragment in str(info.value)


def test_weights_must_match_selected_criteria() -> None:
    with pytest.raises(ConfigError, match="unselected"):
        build_config(_form(criteria=["grammar"], weights={"pace": 2.0}))
    with pytest.raises(ConfigError, match="positive"):
        build_config(_form(criteria=["grammar"], weights={"grammar": 0}))
    config = build_config(_form(criteria=["grammar", "pace"], weights={"grammar": 2.0}))
    assert config.weight_for(EvaluationCategory.GRAMMAR) == 2.0
    assert config.weight_for(EvaluationCategory.PACE) == 1.0


def test_duplicate_criteria_are_removed() -> None:
    config = make_config(criteria=("grammar", "grammar", "pace"))
    assert config.criteria == (EvaluationCategory.GRAMMAR, EvaluationCategory.PACE)


def test_config_is_immutable() -> None:
    config = make_config(consent=ConsentFlags(microphone=True))
    with pytest.raises(ValueError):
        config.duration_min = 10  # type: ignore[misc]


def test_load_job_description_text_and_markdown() -> None:
    assert load_job_description("jd.txt", b"  Senior engineer role  ") == "Senior engineer role"
    assert load_job_description("JD.MD", "\ufeff# Role".encode()) == "# Role"


@pytest.mark.parametrize(
    ("filename", "content", "fragment"),
    [
        ("jd.exe", b"text", "Unsupported"),
        ("jd.txt", b"x" * (MAX_UPLOAD_BYTES + 1), "larger"),
        ("jd.txt", b"\xff\xfe\xfa", "UTF-8"),
        ("jd.txt", b"   \n ", "no text"),
        ("jd.pdf", b"not a pdf", "PDF"),
    ],
    ids=["extension", "too-large", "not-utf8", "empty", "bad-pdf"],
)
def test_load_job_description_rejects_bad_files(
    filename: str, content: bytes, fragment: str
) -> None:
    with pytest.raises(ConfigError, match=fragment):
        load_job_description(filename, content)
