from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import pytest

from train_speaking.providers import torch_runtime

preload = torch_runtime.preload_system_msvc_runtime.__wrapped__


@pytest.fixture
def loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    calls: list[str] = []
    system32 = tmp_path / "windows" / "System32"
    system32.mkdir(parents=True)
    for name in torch_runtime.MSVC_RUNTIME_DLLS:
        (system32 / name).touch()
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path / "windows"))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", calls.append, raising=False)
    return calls


def test_preloads_system_runtime_when_python_bundles_its_own(
    loaded: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = tmp_path / "anaconda"
    base.mkdir()
    (base / "msvcp140.dll").touch()
    monkeypatch.setattr(sys, "base_prefix", str(base))
    preload()
    assert [Path(p).name for p in loaded] == list(torch_runtime.MSVC_RUNTIME_DLLS)
    assert all("System32" in p for p in loaded)


def test_no_preload_for_standard_python(
    loaded: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "python-org"))
    preload()
    assert loaded == []
