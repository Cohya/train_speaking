"""Single entry point for importing torch, with a Windows runtime-DLL workaround."""

from __future__ import annotations

import ctypes
import functools
import os
import sys
from pathlib import Path
from typing import Any

MSVC_RUNTIME_DLLS = ("msvcp140.dll", "vcruntime140_1.dll")


@functools.cache
def preload_system_msvc_runtime() -> None:
    """Load the system's MSVC C++ runtime before torch when Python bundles an older copy.

    Anaconda places MSVC runtime DLLs next to python.exe. They are found before System32,
    and recent torch wheels fail to initialize against them ("WinError 1114 ... c10.dll").
    Once the newer system copies are loaded, the loader reuses them for torch.
    """
    if sys.platform != "win32" or not (Path(sys.base_prefix) / MSVC_RUNTIME_DLLS[0]).is_file():
        return
    system32 = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32"
    for name in MSVC_RUNTIME_DLLS:
        path = system32 / name
        if path.is_file():
            ctypes.WinDLL(str(path))


def import_torch() -> Any:
    preload_system_msvc_runtime()
    import torch

    return torch
