"""Type-check helpers for parser CLI."""

from __future__ import annotations

import shutil
import subprocess
import tempfile

__all__ = ["run_mypy_strict"]


def run_mypy_strict(filename: str) -> tuple[int, str]:
    """Run mypy in strict mode when available; skip otherwise."""
    mypy_path = shutil.which("mypy")
    if mypy_path is None:
        return 0, ""

    try:
        with tempfile.TemporaryDirectory(prefix="esbmc-mypy-cache-") as cache_dir:
            result = subprocess.run(
                [mypy_path, "--strict", "--cache-dir", cache_dir, filename],
                capture_output=True,
                text=True,
                check=False,
            )
    except OSError as exc:
        return 1, f"failed to execute mypy: {exc}"

    output = (result.stdout or "")
    if result.stderr:
        output += result.stderr
    return result.returncode, output
