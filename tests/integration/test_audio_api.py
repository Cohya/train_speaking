from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from train_speaking.api import create_api_router
from train_speaking.models.audio import AudioClip
from train_speaking.session.storage import SessionStorage

TOKEN = "test-api-token"
SESSION_ID = "audio-api-session"


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    storage = SessionStorage(tmp_path / "tmp", tmp_path / "sessions")
    storage.save_answer_audio(SESSION_ID, 3, AudioClip(np.zeros(800, dtype=np.float32), 16_000))
    app = FastAPI()
    app.include_router(create_api_router(storage, TOKEN))
    return TestClient(app)


def test_serves_recorded_audio_with_valid_token(client: TestClient) -> None:
    response = client.get(f"/api/sessions/{SESSION_ID}/audio/turn_3.wav", params={"token": TOKEN})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["cache-control"] == "no-store"
    assert response.content[:4] == b"RIFF"


@pytest.mark.parametrize("token", ["", "wrong-token"])
def test_rejects_missing_or_wrong_token(client: TestClient, token: str) -> None:
    response = client.get(f"/api/sessions/{SESSION_ID}/audio/turn_3.wav", params={"token": token})
    assert response.status_code == 403


@pytest.mark.parametrize(
    ("session_id", "filename"),
    [(SESSION_ID, "turn_9.wav"), ("unknown-session", "turn_3.wav"), (SESSION_ID, "notes.txt")],
)
def test_missing_or_invalid_files_return_404(
    client: TestClient, session_id: str, filename: str
) -> None:
    response = client.get(f"/api/sessions/{session_id}/audio/{filename}", params={"token": TOKEN})
    assert response.status_code == 404
