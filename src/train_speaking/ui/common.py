"""Shared UI helpers."""

from __future__ import annotations

from nicegui import ui

from train_speaking.models.config import CATEGORY_LABELS
from train_speaking.models.session import SessionState

__all__ = ["CATEGORY_LABELS", "STATE_LABELS", "add_assets", "format_seconds", "header"]

STATE_LABELS: dict[SessionState, str] = {
    SessionState.CONFIGURING: "Configuring",
    SessionState.READY: "Ready to start",
    SessionState.INTERVIEWER_SPEAKING: "Interviewer is speaking",
    SessionState.LISTENING: "Listening for your answer",
    SessionState.USER_SPEAKING: "You are speaking",
    SessionState.PROCESSING: "Interviewer is thinking",
    SessionState.PAUSED: "Paused",
    SessionState.RECOVERING: "Recovering from an error",
    SessionState.WRAPPING_UP: "Wrapping up",
    SessionState.EVALUATING: "Preparing your report",
    SessionState.COMPLETED: "Completed",
    SessionState.ABORTED: "Ended",
}


def add_assets() -> None:
    ui.add_head_html('<link rel="stylesheet" href="/static/css/app.css">')


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:d}:{seconds % 60:02d}"


def header(title: str) -> None:
    with ui.row().classes("w-full items-center justify-between"):
        ui.link("Interview Simulator & Speaking Coach", "/").classes("text-h6 no-underline")
        ui.label(title).classes("text-subtitle1 text-grey-7")
