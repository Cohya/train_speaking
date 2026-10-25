"""Session storage: temporary by default, persisted only when the user saves a session.

Layout:
    data/tmp/<session_id>/audio/turn_<id>.wav     (only with recording consent)
    data/sessions/<session_id>/session.json, report.json, report.md, audio/...
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

from train_speaking.models.audio import AudioClip
from train_speaking.models.report import Report
from train_speaking.models.session import Session
from train_speaking.voice.audio_utils import write_wav

logger = logging.getLogger(__name__)

_SESSION_ID = re.compile(r"^[A-Za-z0-9_\-]{8,64}$")
AUDIO_DIR = "audio"


class StorageError(RuntimeError):
    pass


def _check_id(session_id: str) -> str:
    if not _SESSION_ID.fullmatch(session_id):
        raise StorageError("invalid session id")
    return session_id


class SessionStorage:
    def __init__(self, tmp_root: Path, saved_root: Path) -> None:
        self._tmp_root = tmp_root
        self._saved_root = saved_root

    def tmp_dir(self, session_id: str) -> Path:
        return self._tmp_root / _check_id(session_id)

    def saved_dir(self, session_id: str) -> Path:
        return self._saved_root / _check_id(session_id)

    def cleanup_tmp(self) -> None:
        """Remove all unsaved session data (called at startup and shutdown)."""
        if self._tmp_root.exists():
            shutil.rmtree(self._tmp_root)

    def save_answer_audio(self, session_id: str, turn_id: int, clip: AudioClip) -> str:
        filename = f"turn_{turn_id}.wav"
        write_wav(self.tmp_dir(session_id) / AUDIO_DIR / filename, clip.samples, clip.sample_rate)
        return filename

    def audio_path(self, session_id: str, filename: str) -> Path | None:
        if not re.fullmatch(r"turn_\d+\.wav", filename):
            raise StorageError("invalid audio filename")
        for base in (self.saved_dir(session_id), self.tmp_dir(session_id)):
            candidate = base / AUDIO_DIR / filename
            if candidate.is_file():
                return candidate
        return None

    def persist(self, session: Session, report: Report | None, report_markdown: str | None) -> Path:
        target = self.saved_dir(session.id)
        target.mkdir(parents=True, exist_ok=True)
        tmp_audio = self.tmp_dir(session.id) / AUDIO_DIR
        if tmp_audio.is_dir():
            shutil.copytree(tmp_audio, target / AUDIO_DIR, dirs_exist_ok=True)
        (target / "session.json").write_text(session.model_dump_json(indent=2), encoding="utf-8")
        if report is not None:
            (target / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        if report_markdown is not None:
            (target / "report.md").write_text(report_markdown, encoding="utf-8")
        return target

    def is_saved(self, session_id: str) -> bool:
        return (self.saved_dir(session_id) / "session.json").is_file()

    def delete(self, session_id: str) -> None:
        for base in (self.tmp_dir(session_id), self.saved_dir(session_id)):
            if base.exists():
                shutil.rmtree(base)

    def list_saved(self) -> list[Session]:
        sessions: list[Session] = []
        if not self._saved_root.is_dir():
            return sessions
        for path in sorted(self._saved_root.glob("*/session.json")):
            try:
                sessions.append(Session.model_validate_json(path.read_text(encoding="utf-8")))
            except ValueError:
                logger.warning(
                    "skipping unreadable saved session", extra={"path": path.parent.name}
                )
        return sorted(sessions, key=lambda s: s.created_at, reverse=True)

    def load(self, session_id: str) -> tuple[Session, Report | None]:
        directory = self.saved_dir(session_id)
        session_file = directory / "session.json"
        if not session_file.is_file():
            raise StorageError("session not found")
        session = Session.model_validate_json(session_file.read_text(encoding="utf-8"))
        report_file = directory / "report.json"
        report = (
            Report.model_validate(json.loads(report_file.read_text(encoding="utf-8")))
            if report_file.is_file()
            else None
        )
        return session, report
