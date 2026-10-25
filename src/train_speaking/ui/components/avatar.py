"""The virtual interviewer: an original SVG illustration animated by `static/js/avatar.js`."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from nicegui import ui

AVATAR_SVG = Path(__file__).resolve().parents[2] / "static" / "img" / "interviewer.svg"


@cache
def _svg() -> str:
    return AVATAR_SVG.read_text(encoding="utf-8")


def interviewer_avatar() -> None:
    # The SVG is a trusted, bundled asset; user-provided text is never rendered as HTML.
    ui.html(_svg(), sanitize=False).classes("ts-avatar")
