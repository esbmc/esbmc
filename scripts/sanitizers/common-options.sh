# shellcheck shell=bash
# Sourced by sanitizer CI and by local reproduction runs.
#
# Configures ASAN_OPTIONS / UBSAN_OPTIONS / LSAN_OPTIONS so that sanitizer
# diagnostics are written to per-process log files under
# $ESBMC_SANITIZER_LOG_DIR/sanitizer.<pid> instead of stderr.
#
# Why log files?  CTest discards stderr from passing tests, and many
# sanitizer findings live in tests that ESBMC reports as PASSED (the
# instrumentation fires but ESBMC's exit code stays 0).  Logging to files
# survives CTest's filtering and lets collect-findings.sh deduplicate
# across the whole run.
#
# Usage (sourced, not executed):
#     export ESBMC_SANITIZER_LOG_DIR=/path/to/log/dir
#     source scripts/sanitizers/common-options.sh
#
# Resolves $ESBMC_SANITIZER_REPO_ROOT from BASH_SOURCE so callers don't
# have to set it explicitly.

if [ -z "${ESBMC_SANITIZER_LOG_DIR:-}" ]; then
    echo "common-options.sh: ESBMC_SANITIZER_LOG_DIR is unset; defaulting to /tmp" >&2
    ESBMC_SANITIZER_LOG_DIR="/tmp"
fi
mkdir -p "$ESBMC_SANITIZER_LOG_DIR"

# Resolve repo root from this file's location: <repo>/scripts/sanitizers/common-options.sh
_esbmc_san_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ESBMC_SANITIZER_REPO_ROOT="$(cd "$_esbmc_san_dir/../.." && pwd)"
unset _esbmc_san_dir

_log_prefix="$ESBMC_SANITIZER_LOG_DIR/sanitizer"

# Shared across ASan/LSan/UBSan: do not abort the process; do not stop at
# the first error; symbolize stack frames inline; write to log files;
# emit "Suppression: ..." lines so the artifact records which rules
# fired (the collector ignores those lines when ranking findings).
_common="log_path=$_log_prefix:halt_on_error=0:abort_on_error=0:symbolize=1:print_suppressions=1"

ASAN_OPTIONS="$_common:detect_leaks=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1"
UBSAN_OPTIONS="$_common:print_stacktrace=1:report_error_type=1:suppressions=$ESBMC_SANITIZER_REPO_ROOT/scripts/sanitizers/ubsan-suppressions.txt"
LSAN_OPTIONS="$_common:suppressions=$ESBMC_SANITIZER_REPO_ROOT/scripts/sanitizers/lsan-suppressions.txt"

export ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS ESBMC_SANITIZER_LOG_DIR ESBMC_SANITIZER_REPO_ROOT
unset _log_prefix _common
