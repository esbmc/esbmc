#!/usr/bin/env bash
# Verdict-parity sweep: for each regression test, compare the verdict produced
# with the test's own flags (the flag under test OFF) against the verdict with
# it added. Any divergence or flag-on crash is a parity blocker. Deterministic
# verdict comparison per Part V §V.5.
#
# Usage: parity_sweep.sh <esbmc-binary> <regression-subdir> [max-tests]
#
# PARITY_FLAG selects what is being swept; it defaults to V.4.4's
# --irep2-bodies. The frontend adjust-pass hop-offs use the same harness, e.g.
#
#   PARITY_FLAG=--clang-cpp-irep2-adjust-only \
#     scripts/irep2-migration/parity_sweep.sh build/src/esbmc/esbmc \
#     regression/esbmc-cpp/inheritance
#
# which is how docs/roadmap/scope-clang-cpp-irep2.md's Phase 7 counts are taken.
set -u

# Resolve the binary to an absolute path: the sweep cd's into each test dir
# before invoking it, so a relative path would silently fail there (esbmc never
# runs, both sides score OTHER, and the test is falsely reported as parity-OK).
# `command -v` alone is not enough: given a relative path containing a slash it
# succeeds and hands the path back unchanged, so after the cd below esbmc never
# runs, both sides score OTHER, and the sweep reports parity-OK having measured
# nothing. Absolutise a path; leave a bare command name to PATH.
case "$1" in
  */*) ESBMC="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")" ;;
  *)   ESBMC="$(command -v "$1" || printf '%s' "$1")" ;;
esac
if [ ! -x "$ESBMC" ]; then
  echo "error: esbmc binary not found or not executable: $1" >&2
  exit 2
fi
DIR="$2"
MAX="${3:-100000}"
TIMEOUT="${PARITY_TIMEOUT:-30}"
FLAG="${PARITY_FLAG:---irep2-bodies}"

verdict() {
  # Extract a normalized verdict / crash signal from esbmc output.
  if grep -q "VERIFICATION SUCCESSFUL" <<<"$1"; then echo "SUCCESSFUL"
  elif grep -q "VERIFICATION FAILED" <<<"$1"; then echo "FAILED"
  elif grep -q "VERIFICATION UNKNOWN" <<<"$1"; then echo "UNKNOWN"
  elif grep -qi "Segmentation fault\|Assertion .* failed\|terminate called\|what():\|Aborted\|signal SIG\|UNREACHABLE\|terminate\b" <<<"$1"; then echo "CRASH"
  else echo "OTHER"
  fi
}

n=0; diverged=0
while IFS= read -r desc; do
  [ "$n" -ge "$MAX" ] && break
  tdir=$(dirname "$desc")
  # line 1 = CORE/KNOWNBUG/..., line 2 = source, line 3 = flags
  kind=$(sed -n '1p' "$desc")
  src=$(sed -n '2p' "$desc")
  flags=$(sed -n '3p' "$desc")
  [ "$kind" = "KNOWNBUG" ] && continue
  [ -z "$src" ] && continue
  case "$flags" in *"$FLAG"*) continue;; esac   # skip tests already pinning the flag
  n=$((n+1))
  # Split source and flags into words deliberately: $flags is a flag list that
  # must reach esbmc as separate arguments. Arrays keep that splitting explicit
  # (and shellcheck-clean) instead of relying on unquoted expansion.
  read -ra src_arr <<<"$src"
  read -ra flag_arr <<<"$flags"
  out_off=$(cd "$tdir" && timeout "$TIMEOUT" "$ESBMC" "${src_arr[@]}" "${flag_arr[@]}" 2>&1)
  rc_off=$?
  out_on=$(cd "$tdir" && timeout "$TIMEOUT" "$ESBMC" "${src_arr[@]}" "${flag_arr[@]}" "$FLAG" 2>&1)
  rc_on=$?
  v_off=$(verdict "$out_off"); v_on=$(verdict "$out_on")
  # ignore timeouts (rc 124) on either side — not a parity signal
  if [ "$rc_off" = 124 ] || [ "$rc_on" = 124 ]; then continue; fi
  if [ "$v_off" != "$v_on" ]; then
    diverged=$((diverged+1))
    echo "DIVERGE  off=$v_off  on=$v_on  rc_on=$rc_on  $tdir"
  fi
done < <(find "$DIR" -name test.desc |
  { if [ "${PARITY_SHUF:-0}" = 1 ]; then shuf; else sort; fi; })
echo "--- swept $n tests in $DIR; $diverged divergence(s) ---"
