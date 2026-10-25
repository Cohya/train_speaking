"""NiceGUI application assembly."""

from __future__ import annotations

import asyncio
from pathlib import Path

from nicegui import app

from train_speaking.api import create_api_router
from train_speaking.runtime import AppRuntime
from train_speaking.ui.interview_page import register_interview_page
from train_speaking.ui.results_page import register_results_pages
from train_speaking.ui.setup_page import register_setup_page
from train_speaking.voice.ws_endpoint import create_ws_router

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(runtime: AppRuntime) -> None:
    app.add_static_files("/static", STATIC_DIR)
    app.include_router(create_ws_router(runtime.manager))
    app.include_router(create_api_router(runtime.storage, runtime.api_token))
    register_setup_page(runtime)
    register_interview_page(runtime)
    register_results_pages(runtime)

    warmup_tasks: set[asyncio.Task[None]] = set()

    def on_startup() -> None:
        runtime.storage.cleanup_tmp()
        task = asyncio.get_running_loop().create_task(runtime.warmup())
        warmup_tasks.add(task)
        task.add_done_callback(warmup_tasks.discard)

    app.on_startup(on_startup)
    app.on_shutdown(runtime.storage.cleanup_tmp)
