from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from train_speaking.models.config import ConsentFlags, InterviewConfig

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def job_description() -> str:
    return (FIXTURES / "sample_job_description.md").read_text(encoding="utf-8")


def make_config(**overrides: Any) -> InterviewConfig:
    values: dict[str, Any] = {
        "job_description": (FIXTURES / "sample_job_description.md").read_text(encoding="utf-8"),
        "rules": "Ask one question at a time. Ask follow-up questions.",
        "duration_min": 5,
        "consent": ConsentFlags(microphone=True),
    }
    values.update(overrides)
    return InterviewConfig(**values)


@pytest.fixture
def config() -> InterviewConfig:
    return make_config()
