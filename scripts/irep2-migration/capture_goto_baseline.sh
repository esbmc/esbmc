#!/usr/bin/env bash
#
# Capture a golden baseline of canonicalised goto-program dumps for a corpus of
# regression tests. Run this on a clean (pre-migration) build; later migration
# PRs re-run diff_goto_baseline.sh and must produce ZERO diff.
#
# Usage:
#   capture_goto_baseline.sh <esbmc-binary> <baseline-out-dir> <corpus-dir>...
#
# Each <corpus-dir> is searched recursively for `test.desc` files; the source
# and flags from each are used to drive `esbmc --goto-functions-only`.
#
# Example (concurrency + data-race subset, the early-phase scope):
#   scripts/irep2-migration/capture_goto_baseline.sh \
#       build/src/esbmc/esbmc /tmp/irep2-baseline \
#       regression/esbmc/concurrency regression/esbmc-cpp/concurrency
#
# Tracking issue: esbmc/esbmc#4715
set -euo pipefail

# shellcheck source=scripts/irep2-migration/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if [ "$#" -lt 3 ]; then
  sed -n '2,20p' "$0"; exit 2
fi

ESBMC="$(irep2_abspath "$1")"; OUT="$2"; shift 2
REPO="$(irep2_repo_root)"
# Absolutise OUT before we cd, then interpret corpus args relative to repo root.
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
cd "$REPO"

[ -x "$ESBMC" ] || { echo "error: esbmc binary not executable: $ESBMC" >&2; exit 2; }

manifest="$OUT/MANIFEST.sha256"
: > "$manifest"
count=0

for corpus in "$@"; do
  [ -d "$corpus" ] || { echo "warning: skipping missing corpus dir: $corpus" >&2; continue; }
  while IFS= read -r desc; do
    dir="$(dirname "$desc")"
    # Stable per-test key: repo-relative dir, slashes -> _.
    key="$(printf '%s' "$dir" | tr '/' '_')"
    dump="$OUT/$key.goto.txt"

    irep2_goto_dump "$ESBMC" "$desc" "$REPO" > "$dump" || { rm -f "$dump"; continue; }
    ( cd "$OUT" && shasum -a 256 "$key.goto.txt" ) >> "$manifest"
    count=$((count + 1))
  done < <(find "$corpus" -name test.desc | sort)
done

irep2_cleanup_tmp
sort -o "$manifest" "$manifest"
echo "captured $count goto dumps into $OUT"
echo "manifest: $manifest"
