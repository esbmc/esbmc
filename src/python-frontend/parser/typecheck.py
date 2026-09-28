"""Type-check helpers for parser CLI."""

from __future__ import annotations

import re
import tempfile

try:
    from mypy import api as mypy_api
except ImportError:  # pragma: no cover - environment-dependent
    mypy_api = None

__all__ = ["run_mypy_strict"]

# Verification intrinsics are supplied by the frontend rather than declared in
# the source, so mypy reports every use as undefined. Matched by exact name: a
# prefix wildcard would also swallow a misspelling such as `__ESBMC_assme`,
# which is the very thing the report should still show.
_INTRINSICS = (
    "__ESBMC_assume",
    "__ESBMC_assert",
    "__ESBMC_requires",
    "__ESBMC_ensures",
    "__ESBMC_return_value",
    "__ESBMC_old",
    "__ESBMC_assigns",
    "__ESBMC_is_fresh",
    "__ESBMC_forall",
    "__ESBMC_exists",
    "__ESBMC_unreachable",
    "__ESBMC_cover",
    "__ESBMC_loop_invariant",
    "__loop_invariant",
    "__ESBMC_atomic_begin",
    "__ESBMC_atomic_end",
    "__VERIFIER_assume",
    "__VERIFIER_error",
    "nondet_int",
    "nondet_uint",
    "nondet_long",
    "nondet_ulong",
    "nondet_short",
    "nondet_ushort",
    "nondet_char",
    "nondet_uchar",
    "nondet_float",
    "nondet_double",
    "nondet_bool",
    "nondet_str",
    "nondet_size",
)
_INTRINSIC_UNDEFINED = re.compile('error: Name "(?:' +
                                  "|".join(re.escape(name)
                                           for name in _INTRINSICS) + ')" is not defined')


def _run_mypy_module(filename: str, cache_dir: str) -> tuple[int, str]:
    """Run mypy through ``mypy.api``."""
    report, errors, exit_status = mypy_api.run(  # pylint: disable=c-extension-no-member
        ["--strict", "--cache-dir", cache_dir, filename])
    return int(exit_status), f"{report}{errors}"


def _without_intrinsic_diagnostics(exit_status: int, report: str) -> tuple[int, str]:
    """Drop undefined-name diagnostics for ESBMC's injected globals."""
    lines = report.splitlines()
    kept = [line for line in lines if not _INTRINSIC_UNDEFINED.search(line)]
    # Nothing suppressed: pass the report through untouched, so a mypy failure
    # that formats without a ": error:" line (a missing file, an internal
    # error) still reaches the caller.
    if len(kept) == len(lines):
        return exit_status, report
    if not any(": error:" in line for line in kept):
        return 0, ""
    return exit_status, "\n".join(line for line in kept if not line.startswith("Found "))


def run_mypy_strict(filename: str) -> tuple[int, str]:
    """Run mypy in strict mode when the Python module is available."""
    if mypy_api is None:
        return 0, ""

    with tempfile.TemporaryDirectory(prefix="esbmc-mypy-cache-") as cache_dir:
        return _without_intrinsic_diagnostics(*_run_mypy_module(filename, cache_dir))
