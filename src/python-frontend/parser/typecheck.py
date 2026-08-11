"""Type-check helpers for parser CLI."""

from __future__ import annotations

import re
import tempfile

try:
    from mypy import api as mypy_api
except ImportError:  # pragma: no cover - environment-dependent
    mypy_api = None

__all__ = ["run_mypy_strict"]

# Verification intrinsics (__ESBMC_assume, nondet_int, __ESBMC_requires, ...)
# are supplied by the frontend rather than declared in the source, so mypy
# reports every use as undefined. Suppress only those; a name-defined error
# for a name the user actually mistyped still reaches the report.
_INTRINSIC_UNDEFINED = re.compile(
    r'error: Name "(__ESBMC_\w+|__VERIFIER_\w+|nondet_\w+|__loop_invariant)"'
    r' is not defined')


def _run_mypy_module(filename: str, cache_dir: str) -> tuple[int, str] | None:
    """Run mypy through ``mypy.api`` when importable."""
    if mypy_api is None:
        return None

    report, errors, exit_status = mypy_api.run(  # pylint: disable=c-extension-no-member
        ["--strict", "--cache-dir", cache_dir, filename])
    return int(exit_status), f"{report}{errors}"


def _without_intrinsic_diagnostics(exit_status: int, report: str) -> tuple[int, str]:
    """Drop undefined-name diagnostics for ESBMC's injected globals."""
    kept = [line for line in report.splitlines() if not _INTRINSIC_UNDEFINED.search(line)]
    if not any(": error:" in line for line in kept):
        return 0, ""
    return exit_status, "\n".join(line for line in kept if not line.startswith("Found "))


def run_mypy_strict(filename: str) -> tuple[int, str]:
    """Run mypy in strict mode when the Python module is available."""
    if mypy_api is None:
        return 0, ""

    with tempfile.TemporaryDirectory(prefix="esbmc-mypy-cache-") as cache_dir:
        module_result = _run_mypy_module(filename, cache_dir)
        if module_result is None:
            return 0, ""
        return _without_intrinsic_diagnostics(*module_result)
