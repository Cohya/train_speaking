"""Exposes the runtime built by `app_main.py` to the UI tests."""

from train_speaking.runtime import AppRuntime

runtime: AppRuntime | None = None
