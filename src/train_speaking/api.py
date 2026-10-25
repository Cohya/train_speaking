"""HTTP routes for recorded answer audio."""

from __future__ import annotations

import secrets

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from train_speaking.session.storage import SessionStorage, StorageError


def create_api_router(storage: SessionStorage, api_token: str) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/sessions/{session_id}/audio/{filename}")
    def answer_audio(session_id: str, filename: str, token: str = "") -> FileResponse:
        if not secrets.compare_digest(token, api_token):
            raise HTTPException(status_code=403)
        try:
            path = storage.audio_path(session_id, filename)
        except StorageError as exc:
            raise HTTPException(status_code=404) from exc
        if path is None:
            raise HTTPException(status_code=404)
        return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    return router
