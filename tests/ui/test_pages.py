from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from nicegui import ui
from nicegui.testing.user import User
from nicegui.testing.user_simulation import user_simulation

from tests.conftest import make_config
from tests.ui import app_holder
from train_speaking.evaluation.evaluator import SpeakingEvaluator
from train_speaking.models.session import SessionState
from train_speaking.models.transcript import Role, Turn
from train_speaking.runtime import AppRuntime

MAIN_FILE = Path(__file__).parent / "app_main.py"
SAMPLE_JD = (Path(__file__).parents[1] / "fixtures" / "sample_job_description.md").read_text(
    encoding="utf-8"
)


@pytest.fixture
async def ui_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncIterator[User]:
    for name, value in {
        "TS_DATA_DIR": str(tmp_path / "data"),
        "TS_MODELS_DIR": str(tmp_path / "models"),
        "TS_STT_PROVIDER": "fake",
        "TS_TTS_PROVIDER": "fake",
        "TS_VAD_PROVIDER": "energy",
        "TS_LLM_PROVIDER": "none",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(tmp_path)
    async with user_simulation(main_file=MAIN_FILE) as user:
        yield user


def _runtime() -> AppRuntime:
    assert app_holder.runtime is not None
    return app_holder.runtime


async def test_setup_page_rejects_missing_job_description(ui_user: User) -> None:
    await ui_user.open("/")
    await ui_user.should_see("Set up your interview")
    await ui_user.should_see("No LLM is configured")
    ui_user.find(marker="consent-microphone").click()
    ui_user.find("Start interview").click()
    await ui_user.should_see("job description")
    assert _runtime().manager.get_all() == []


async def test_setup_page_requires_microphone_consent(ui_user: User) -> None:
    await ui_user.open("/")
    ui_user.find(marker="job-description").type(SAMPLE_JD)
    ui_user.find("Start interview").click()
    await ui_user.should_see("microphone consent is required")
    assert _runtime().manager.get_all() == []


async def test_example_rules_button_fills_rules(ui_user: User) -> None:
    await ui_user.open("/")
    ui_user.find("Use example rules").click()
    rules = ui_user.find(marker="rules").elements.pop()
    assert isinstance(rules, ui.textarea)
    assert "Ask one question at a time." in str(rules.value)


async def test_valid_setup_creates_session_and_opens_interview(ui_user: User) -> None:
    await ui_user.open("/")
    ui_user.find(marker="job-description").type(SAMPLE_JD)
    ui_user.find(marker="consent-microphone").click()
    ui_user.find("Start interview").click()
    await ui_user.should_see("Press Start when you are ready.")
    sessions = _runtime().manager.get_all()
    assert len(sessions) == 1
    assert sessions[0].session.state is SessionState.READY
    assert sessions[0].session.config.consent.microphone


async def test_results_page_renders_report_with_basis_labels(ui_user: User) -> None:
    runtime = _runtime()
    session_runtime = runtime.manager.create(make_config())
    session = session_runtime.session
    session.turns = [
        Turn(id=1, role=Role.INTERVIEWER, text="Tell me about yourself.", started_at=0,
             ended_at=1, topic_id="opening"),
        Turn(id=2, role=Role.CANDIDATE, text="I build reliable backend services.", started_at=2,
             ended_at=5, is_text_answer=True),
    ]  # fmt: skip
    session.state = SessionState.COMPLETED
    session_runtime.orchestrator.report = await SpeakingEvaluator(None).evaluate(session)

    await ui_user.open(f"/results/{session.id}")
    await ui_user.should_see("Your interview report")
    await ui_user.should_see("Methodology and limitations")
    await ui_user.should_see("Not assessed")
    await ui_user.should_see("A: I build reliable backend services.")
    ui_user.find("Save session").click()
    await ui_user.should_see("Session saved")
    assert runtime.storage.is_saved(session.id)


async def test_unknown_results_page_explains_missing_session(ui_user: User) -> None:
    await ui_user.open("/results/doesnotexist123")
    await ui_user.should_see("This session was not found")
