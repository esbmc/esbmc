#!/usr/bin/env bash
#
# Count legacy<->IREP2 conversion calls (migrate_expr, migrate_type,
# migrate_expr_back, migrate_type_back) per file. This is the migration
# "scoreboard": each migration PR must show the count strictly DECREASING in the
# region it targets and increasing nowhere else (the Phase-0 measurable
# checkpoint from docs/irep2-goto-migration-plan.md §6).
#
# Usage:
#   migrate_census.sh [path ...]      # default: src/goto-programs
#
# Output: per-file counts (forward / back / total) sorted by total descending,
# then a grand total. Stable and greppable, suitable for checking a snapshot
# into the tree and diffing across PRs.
#
# Tracking issue: esbmc/esbmc#4715
set -u

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo" || exit 1

paths=("$@")
[ "${#paths[@]}" -gt 0 ] || paths=("src/goto-programs")

# Forward = into IREP2 (migrate_expr / migrate_type). The _back variants also
# match this pattern, so we subtract them out below.
fwd_re='migrate_(expr|type)\b'
back_re='migrate_(expr|type)_back\b'

# Count occurrences of an ERE in a file (0 on no match; never fails the script).
count() { grep -E -o "$1" "$2" 2>/dev/null | wc -l | tr -d ' '; }

rows=""
tot_fwd=0; tot_back=0

while IFS= read -r f; do
  back=$(count "$back_re" "$f")
  all=$(count "$fwd_re" "$f")
  fwd=$(( all - back ))
  total=$(( fwd + back ))
  [ "$total" -gt 0 ] || continue
  rows+="$(printf '%6d\t%6d\t%6d\t%s\n' "$total" "$fwd" "$back" "${f#"$repo"/}")"$'\n'
  tot_fwd=$(( tot_fwd + fwd )); tot_back=$(( tot_back + back ))
done < <(
  for p in "${paths[@]}"; do
    find "$p" -type f \( -name '*.cpp' -o -name '*.h' \) 2>/dev/null
  done | sort
)

printf '%-58s %6s %6s %6s\n' "FILE" "FWD" "BACK" "TOTAL"
printf '%-58s %6s %6s %6s\n' "----" "---" "----" "-----"
# Sort rows by total (col 1) descending, then re-layout as FILE FWD BACK TOTAL.
printf '%s' "$rows" | sort -t$'\t' -k1,1nr -k4,4 | while IFS=$'\t' read -r total fwd back file; do
  [ -n "$file" ] || continue
  printf '%-58s %6d %6d %6d\n' "$file" "$fwd" "$back" "$total"
done
printf '%-58s %6s %6s %6s\n' "----" "---" "----" "-----"
printf '%-58s %6d %6d %6d\n' "TOTAL" "$tot_fwd" "$tot_back" "$(( tot_fwd + tot_back ))"
