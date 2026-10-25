from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import make_config
from train_speaking.evaluation.evaluator import SpeakingEvaluator
from train_speaking.models.audio import AudioClip
from train_speaking.models.session import Session, SessionState
from train_speaking.models.transcript import Role, Turn
from train_speaking.session.storage import SessionStorage, StorageError

SESSION_ID = "storage-session-1"


@pytest.fixture
def storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage(tmp_path / "tmp", tmp_path / "sessions")


def _session(session_id: str = SESSION_ID) -> Session:
    return Session(
        id=session_id,
        config=make_config(),
        state=SessionState.COMPLETED,
        turns=[
            Turn(id=1, role=Role.INTERVIEWER, text="Why this role?", started_at=0, ended_at=1,
                 topic_id="opening"),
            Turn(id=2, role=Role.CANDIDATE, text="It matches my backend experience.",
                 started_at=2, ended_at=4, audio_file="turn_2.wav"),
        ],
    )  # fmt: skip


def _clip() -> AudioClip:
    return AudioClip(np.zeros(1600, dtype=np.float32), 16_000)


def test_answer_audio_is_temporary_until_saved(storage: SessionStorage) -> None:
    filename = storage.save_answer_audio(SESSION_ID, 2, _clip())
    assert filename == "turn_2.wav"
    path = storage.audio_path(SESSION_ID, filename)
    assert path is not None and path.is_file()
    assert not storage.is_saved(SESSION_ID)
    storage.cleanup_tmp()
    assert storage.audio_path(SESSION_ID, filename) is None


async def test_persist_and_load_round_trip(storage: SessionStorage) -> None:
    session = _session()
    storage.save_answer_audio(SESSION_ID, 2, _clip())
    report = await SpeakingEvaluator(None).evaluate(session)
    target = storage.persist(session, report, "# Interview report")
    assert {p.name for p in target.iterdir()} == {
        "session.json",
        "report.json",
        "report.md",
        "audio",
    }

    storage.cleanup_tmp()
    loaded, loaded_report = storage.load(SESSION_ID)
    assert loaded.turns == session.turns
    assert loaded_report is not None and loaded_report.session_id == SESSION_ID
    assert storage.audio_path(SESSION_ID, "turn_2.wav") is not None
    assert [s.id for s in storage.list_saved()] == [SESSION_ID]


def test_persist_without_report(storage: SessionStorage) -> None:
    storage.persist(_session(), None, None)
    _, report = storage.load(SESSION_ID)
    assert report is None


def test_delete_removes_saved_and_temporary_data(storage: SessionStorage) -> None:
    storage.save_answer_audio(SESSION_ID, 2, _clip())
    storage.persist(_session(), None, None)
    storage.delete(SESSION_ID)
    assert not storage.is_saved(SESSION_ID)
    assert storage.audio_path(SESSION_ID, "turn_2.wav") is None
    with pytest.raises(StorageError):
        storage.load(SESSION_ID)


def test_list_saved_skips_corrupt_sessions(storage: SessionStorage, tmp_path: Path) -> None:
    storage.persist(_session(), None, None)
    corrupt = tmp_path / "sessions" / "corrupt-session" / "session.json"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("{not json", encoding="utf-8")
    assert [s.id for s in storage.list_saved()] == [SESSION_ID]


@pytest.mark.parametrize("bad_id", ["../escape", "short", "a/b/c/d/e/f", "x" * 65, "..\\..\\win"])
def test_rejects_unsafe_session_ids(storage: SessionStorage, bad_id: str) -> None:
    with pytest.raises(StorageError):
        storage.tmp_dir(bad_id)


@pytest.mark.parametrize(
    "bad_name", ["../session.json", "turn_1.wav/../../x", "turn_a.wav", "x.wav"]
)
def test_rejects_unsafe_audio_filenames(storage: SessionStorage, bad_name: str) -> None:
    with pytest.raises(StorageError):
        storage.audio_path(SESSION_ID, bad_name)
