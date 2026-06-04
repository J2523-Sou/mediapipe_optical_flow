"""Runtime delegate selection helpers for MediaPipe Tasks."""

from __future__ import annotations

import platform


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def should_try_gpu(prefer_gpu: bool) -> bool:
    """Return whether it is safe to attempt MediaPipe Tasks GPU delegate."""
    return prefer_gpu and not is_apple_silicon()


def cpu_delegate_name() -> str:
    if is_apple_silicon():
        return "CPU (Apple Silicon)"
    return "CPU"


def explain_cpu_fallback(prefer_gpu: bool) -> None:
    if prefer_gpu and is_apple_silicon():
        print(
            "Apple Silicon detected: using CPU delegate. "
            "MediaPipe Tasks GPU delegate can abort the Python process on macOS arm64."
        )
