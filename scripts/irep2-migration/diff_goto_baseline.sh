#!/usr/bin/env bash
#
# Re-derive canonical goto dumps for a corpus and diff them against a baseline
# captured by capture_goto_baseline.sh. The migration gate: a behaviour-
# preserving PR must produce ZERO diff. Any difference is reported per-test and
# the script exits non-zero.
#
# Usage:
#   diff_goto_baseline.sh <esbmc-binary> <baseline-dir> <corpus-dir>...
#
# Exit codes: 0 = identical to baseline, 1 = differences found, 2 = usage error.
#
# Tracking issue: esbmc/esbmc#4715
set -euo pipefail

# shellcheck source=scripts/irep2-migration/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if [ "$#" -lt 3 ]; then
  sed -n '2,16p' "$0"; exit 2
fi

ESBMC="$(irep2_abspath "$1")"; BASE="$2"; shift 2
REPO="$(irep2_repo_root)"
[ -d "$BASE" ] && BASE="$(cd "$BASE" && pwd)"
cd "$REPO"

[ -x "$ESBMC" ] || { echo "error: esbmc binary not executable: $ESBMC" >&2; exit 2; }
[ -d "$BASE" ]  || { echo "error: baseline dir not found: $BASE" >&2; exit 2; }

work="$(mktemp -d)"
trap 'rm -rf "$work"; irep2_cleanup_tmp' EXIT

differing=0; checked=0; missing=0

for corpus in "$@"; do
  [ -d "$corpus" ] || { echo "warning: skipping missing corpus dir: $corpus" >&2; continue; }
  while IFS= read -r desc; do
    dir="$(dirname "$desc")"
    key="$(printf '%s' "$dir" | tr '/' '_')"
    golden="$BASE/$key.goto.txt"

    irep2_goto_dump "$ESBMC" "$desc" "$REPO" > "$work/cur.txt" || continue
    if [ ! -f "$golden" ]; then
      echo "MISSING-BASELINE  $dir"; missing=$((missing + 1)); continue
    fi
    if ! diff -u "$golden" "$work/cur.txt" > "$work/d.txt"; then
      echo "DIFF              $dir"
      sed 's/^/    /' "$work/d.txt"
      differing=$((differing + 1))
    fi
    checked=$((checked + 1))
  done < <(find "$corpus" -name test.desc | sort)
done

echo "checked=$checked  differing=$differing  missing-baseline=$missing"
[ "$differing" -eq 0 ] && [ "$missing" -eq 0 ]
