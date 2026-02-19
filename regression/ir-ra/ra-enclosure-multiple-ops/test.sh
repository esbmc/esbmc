#!/usr/bin/env bash
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$TEST_DIR"

KEEP_TMP="${KEEP_TMP:-0}"

REPO_ROOT="$(cd "$TEST_DIR/../../.." && pwd)"

if [[ -n "${ESBMC:-}" ]]; then
  if [[ "$ESBMC" = /* ]]; then
    ESBMC_BIN="$ESBMC"
  else
    ESBMC_BIN="$REPO_ROOT/$ESBMC"
  fi
else
  ESBMC_BIN="$REPO_ROOT/build/src/esbmc/esbmc"
fi

if [[ ! -x "$ESBMC_BIN" ]]; then
  echo "ERROR: ESBMC binary not found or not executable: $ESBMC_BIN" >&2
  exit 1
fi

TMP_RAW="$(mktemp -t irra-XXXXXX.raw.smt2)"
TMP_SMT="$(mktemp -t irra-XXXXXX.smt2)"
TMP_LOG="$(mktemp -t irra-XXXXXX.log)"

cleanup() {
  status=$?
  if [[ "$status" -eq 0 && "$KEEP_TMP" != "1" ]]; then
    rm -f "$TMP_RAW" "$TMP_SMT" "$TMP_LOG"
  else
    echo "Keeping debug artifacts:" >&2
    echo "  RAW: $TMP_RAW" >&2
    echo "  SMT: $TMP_SMT" >&2
    echo "  LOG: $TMP_LOG" >&2
  fi
  exit "$status"
}
trap cleanup EXIT

"$ESBMC_BIN" main.c --ir-ra --smtlib --smt-formula-only --output "$TMP_RAW" >"$TMP_LOG" 2>&1 || true

if ! grep -q "Encoding remaining VCC(s) using integer/real arithmetic" "$TMP_LOG"; then
  echo "ERROR: --ir-ra did not run IR encoding" >&2
  tail -n 120 "$TMP_LOG" >&2
  exit 1
fi

tr -d '\000' < "$TMP_RAW" > "$TMP_SMT"

LO_COUNT="$(grep -o "smt_conv::ra_lo::" "$TMP_SMT" | wc -l | tr -d ' ')"
HI_COUNT="$(grep -o "smt_conv::ra_hi::" "$TMP_SMT" | wc -l | tr -d ' ')"

if [[ "$LO_COUNT" -ge 2 && "$HI_COUNT" -ge 2 ]]; then
  exit 0
fi

echo "ERROR: expected at least 2 ra_lo and 2 ra_hi, got lo=$LO_COUNT hi=$HI_COUNT" >&2
tail -n 120 "$TMP_LOG" >&2
exit 1
