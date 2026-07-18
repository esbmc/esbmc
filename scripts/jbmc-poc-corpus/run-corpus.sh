#!/usr/bin/env bash
#
# Run every corpus program through jbmc-poc-pipeline.sh and print one row per
# program: JBMC's own verdict, ESBMC's, and whether they agree.
#
#   PATH=/path/to/jdk/bin:$PATH ESBMC=/path/to/esbmc ./run-corpus.sh
#
# Programs ending in _fail carry a violated property; the rest verify clean.
# JBMC's verdicts are the reference (README.md records the measured values), so
# a disagreement is an ESBMC finding, not a corpus defect.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIPELINE="$HERE/../jbmc-poc-pipeline.sh"
OUTROOT="${OUTROOT:-jbmc-poc-corpus-out}"
UNWIND="${UNWIND:-6}"
ESBMC="${ESBMC:-esbmc}"

[ -x "$PIPELINE" ] || { echo "error: $PIPELINE not found" >&2; exit 1; }

# Reused by the pipeline; resolved here too so a missing jar fails before the
# first program rather than ten times over.
MODELS="${CORE_MODELS_JAR:-}"
if [ -z "$MODELS" ]; then
  for c in /usr/lib/core-models.jar \
           /opt/homebrew/Cellar/cbmc/*/libexec/lib/core-models.jar \
           /usr/local/Cellar/cbmc/*/libexec/lib/core-models.jar; do
    [ -f "$c" ] && { MODELS="$c"; break; }
  done
fi
[ -n "$MODELS" ] && [ -f "$MODELS" ] || { echo "error: core-models.jar not found" >&2; exit 1; }

# grep exits 1 when a run produced no verdict at all (ESBMC declining, say).
# Under `set -e` with pipefail that would abort the sweep on the first such
# program, which is exactly the case this table exists to report.
verdict_of() {
  { grep -oE 'VERIFICATION (SUCCESSFUL|FAILED)' "$1" || true; } 2>/dev/null \
    | tail -1 | sed 's/VERIFICATION //'
}

mkdir -p "$OUTROOT"
printf '%-20s %-12s %-12s %-12s %s\n' PROGRAM EXPECTED JBMC ESBMC AGREE
agree=0; total=0

for src in "$HERE"/T*.java; do
  cls=$(basename "$src" .java)
  case "$cls" in *_fail) expected=FAILED ;; *) expected=SUCCESSFUL ;; esac

  work="$OUTROOT/$cls"
  mkdir -p "$work"

  # JBMC reference verdict, compiled and run from a scratch dir.
  javac -d "$work" "$src" 2>/dev/null || { echo "$cls: javac failed" >&2; continue; }
  jbmc "$cls" -cp "$work:$MODELS" --unwind "$UNWIND" > "$work/jbmc-verdict.log" 2>&1 || true
  jv=$(verdict_of "$work/jbmc-verdict.log")

  # ESBMC via the shared pipeline.
  ESBMC="$ESBMC" "$PIPELINE" --class "$cls" --source "$src" --lazy \
    --outdir "$work/pipeline" > /dev/null 2>&1 || true
  ev=$(verdict_of "$work/pipeline/esbmc.log")
  if [ -z "$ev" ]; then
    ev=$(grep -m1 '^ERROR:' "$work/pipeline/esbmc.log" 2>/dev/null \
           | cut -c1-40 || true)
    ev="${ev:-no-verdict}"
  fi

  total=$((total + 1))
  if [ "$jv" = "$ev" ]; then mark=yes; agree=$((agree + 1)); else mark=no; fi
  printf '%-20s %-12s %-12s %-12s %s\n' "$cls" "$expected" "${jv:-none}" "$ev" "$mark"
done

echo
echo "ESBMC agrees with JBMC on $agree/$total corpus programs."
