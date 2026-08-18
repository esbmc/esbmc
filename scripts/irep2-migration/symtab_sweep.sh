#!/usr/bin/env bash
#
# Compare the symbol table an adjust pass produces with and without a flag, over
# a corpus of regression tests. This is the instrument for adjuster questions:
# the symbol table is what an adjust pass writes, whereas a goto program is two
# stages further on and mixes in goto_convert's own behaviour. See
# docs/roadmap/scope-clang-c-irep2.md §100.1.
#
# Usage:
#   symtab_sweep.sh <esbmc-binary> <flag> <corpus-dir>...
#
# Example (the Phase 6 hop-off):
#   scripts/irep2-migration/symtab_sweep.sh \
#       build/src/esbmc/esbmc --clang-c-irep2-adjust-only regression/esbmc
#
# Prints one line per test (SAME / DIFF / SKIP) and a total. Blank-line-only
# differences are ignored: the symbol-table printer varies its blank lines with
# block nesting, which is not a difference in what was adjusted.
#
# Tracking issue: esbmc/esbmc#4715
set -u

# shellcheck source=scripts/irep2-migration/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if [ "$#" -lt 3 ]; then
  sed -n '2,20p' "$0"; exit 2
fi

ESBMC="$(irep2_abspath "$1")"; FLAG="$2"; shift 2
REPO="$(irep2_repo_root)"
cd "$REPO"

[ -x "$ESBMC" ] || { echo "error: esbmc binary not executable: $ESBMC" >&2; exit 2; }

same=0; diff_count=0; skipped=0

for corpus in "$@"; do
  [ -d "$corpus" ] || { echo "warning: skipping missing corpus dir: $corpus" >&2; continue; }
  while IFS= read -r desc; do
    dir="$(dirname "$desc")"

    if ! off="$(irep2_symtab_dump "$ESBMC" "$desc" "$REPO")"; then
      echo "SKIP $dir"; skipped=$((skipped + 1)); continue
    fi

    on="$(irep2_symtab_dump "$ESBMC" "$desc" "$REPO" "$FLAG")"

    if diff -B <(printf '%s\n' "$off") <(printf '%s\n' "$on") > /dev/null; then
      echo "SAME $dir"; same=$((same + 1))
    else
      echo "DIFF $dir"; diff_count=$((diff_count + 1))
    fi
  done < <(find "$corpus" -name test.desc | sort)
done

irep2_cleanup_tmp
echo "--- $same same, $diff_count differing, $skipped skipped ---"
