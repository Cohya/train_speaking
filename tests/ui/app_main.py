"""NiceGUI entry point for UI simulation tests; settings come from TS_* variables set by tests."""

from nicegui import ui

from tests.ui import app_holder
from train_speaking.app import create_app
from train_speaking.runtime import AppRuntime
from train_speaking.settings import AppSettings

app_holder.runtime = AppRuntime(AppSettings())
create_app(app_holder.runtime)
ui.run(reload=False, show=False)
