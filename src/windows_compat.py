"""Windows compatibility helpers for optional FluidSynth support."""

from __future__ import annotations

import os


_INSTALLED = False


class _IgnoredDllDirectory:
    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def ignore_missing_optional_fluidsynth_path() -> None:
    """Let pretty_midi import when pyFluidSynth's optional Windows DLL path is absent."""
    global _INSTALLED
    if _INSTALLED or not hasattr(os, "add_dll_directory"):
        return

    real_add_dll_directory = os.add_dll_directory
    optional_path = os.path.normcase(os.path.normpath(r"C:\tools\fluidsynth\bin"))

    def add_dll_directory(path: str):
        try:
            return real_add_dll_directory(path)
        except FileNotFoundError:
            normalized = os.path.normcase(os.path.normpath(path))
            if normalized == optional_path:
                return _IgnoredDllDirectory()
            raise

    os.add_dll_directory = add_dll_directory
    _INSTALLED = True
