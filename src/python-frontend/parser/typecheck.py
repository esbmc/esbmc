"""Type-check helpers for parser CLI."""

from __future__ import annotations

import tempfile

try:
    from mypy import api as mypy_api
except ImportError:  # pragma: no cover - environment-dependent
    mypy_api = None

__all__ = ["run_mypy_strict"]


def _run_mypy_module(filename: str, cache_dir: str) -> tuple[int, str] | None:
    """Run mypy through ``mypy.api`` when importable."""
    if mypy_api is None:
        return None

    report, errors, exit_status = mypy_api.run(  # pylint: disable=c-extension-no-member
        ["--strict", "--cache-dir", cache_dir, filename]
    )
    return int(exit_status), f"{report}{errors}"


def run_mypy_strict(filename: str) -> tuple[int, str]:
    """Run mypy in strict mode when the Python module is available."""
    if mypy_api is None:
        return 0, ""

    with tempfile.TemporaryDirectory(prefix="esbmc-mypy-cache-") as cache_dir:
        module_result = _run_mypy_module(filename, cache_dir)
        if module_result is not None:
            return module_result
        return 0, ""
