"""Entry point: `python -m train_speaking`."""

from __future__ import annotations

import os

from train_speaking.settings import AppSettings


def main() -> None:
    settings = AppSettings()
    # Keep downloaded model weights inside the project (set for this process only, before
    # any Hugging Face library is imported).
    os.environ.setdefault("HF_HOME", str(settings.models_dir / "huggingface"))

    from nicegui import ui

    from train_speaking.app import create_app
    from train_speaking.logging_setup import configure_logging
    from train_speaking.runtime import AppRuntime

    configure_logging(settings.log_level)
    create_app(AppRuntime(settings))
    ui.run(
        host=settings.host,
        port=settings.port,
        title="Interview Simulator",
        reload=False,
        show=settings.open_browser,
        show_welcome_message=True,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
