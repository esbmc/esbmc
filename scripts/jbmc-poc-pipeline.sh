#!/usr/bin/env bash
#
# Run the JBMC -> goto-binary -> ESBMC pipeline described in
# docs/jbmc-goto-binary-poc-plan.md §2.2, and record everything a later reader
# needs to reproduce the result.
#
#   scripts/jbmc-poc-pipeline.sh --class java.lang.Integer
#   scripts/jbmc-poc-pipeline.sh --source T1Arith.java --class T1Arith
#
# Every run writes <outdir>/manifest.txt with the tool versions, the resolved
# models jar and the ESBMC verdict, because a verdict from this pipeline is
# meaningful only relative to the classpath that produced the model (§2.2: the
# function count swings ~48x on flags alone).

set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

# macOS ships a javac stub that exists on PATH and exits 0 while reporting "no
# Java runtime" on stderr, so `command -v javac` is not a usable probe. Match
# the version string instead.
javac_version() {
  local v
  v=$(javac -version 2>&1 | head -1)
  case "$v" in
    javac\ *) echo "$v" ;;
    *) echo "not present" ;;
  esac
}

CLASS=""
SOURCE=""
OUTDIR="jbmc-poc-out"
ESBMC="${ESBMC:-esbmc}"
ESBMC_ARGS=()

usage() {
  cat >&2 <<EOF
usage: $0 --class <fully.qualified.Class> [--source <File.java>]
          [--outdir <dir>] [-- <extra esbmc args>]

  --class   class to verify; required. Taken from core-models.jar unless
            --source is given.
  --source  Java source to compile first. Requires a JDK on PATH.
  --outdir  where artefacts land (default: $OUTDIR).
EOF
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --class)  CLASS="${2:-}"; shift 2 ;;
    --source) SOURCE="${2:-}"; shift 2 ;;
    --outdir) OUTDIR="${2:-}"; shift 2 ;;
    --)       shift; ESBMC_ARGS=("$@"); break ;;
    -h|--help) usage ;;
    *) die "unknown argument '$1' (try --help)" ;;
  esac
done

[ -n "$CLASS" ] || usage

for tool in jbmc cbmc symtab2gb python3; do
  command -v "$tool" >/dev/null || die "'$tool' not found on PATH"
done
command -v "$ESBMC" >/dev/null || die "esbmc not found (set ESBMC=/path/to/esbmc)"

# core-models.jar is not at a portable path and is never loaded by default
# (§2.2). Without it jbmc silently emits a ~50-function model instead of
# failing, so an unfound jar has to be fatal rather than a warning.
MODELS="${CORE_MODELS_JAR:-}"
if [ -z "$MODELS" ]; then
  for candidate in \
      /usr/lib/core-models.jar \
      /opt/homebrew/Cellar/cbmc/*/libexec/lib/core-models.jar \
      /usr/local/Cellar/cbmc/*/libexec/lib/core-models.jar \
      /usr/lib/cbmc/core-models.jar; do
    [ -f "$candidate" ] && { MODELS="$candidate"; break; }
  done
fi
# Validate the path itself, not just that one was chosen: a typo'd
# CORE_MODELS_JAR would otherwise reach jbmc, which loads what it can and
# reports success on a model missing most of its method bodies.
[ -n "$MODELS" ] || die "core-models.jar not found; a model built without it \
is ~50 functions and any verdict from it is meaningless (see plan §2.2). \
Set CORE_MODELS_JAR to override."
[ -f "$MODELS" ] || die "core-models.jar '$MODELS' does not exist"

mkdir -p "$OUTDIR"
CP="$MODELS"

if [ -n "$SOURCE" ]; then
  [ "$(javac_version)" != "not present" ] \
    || die "--source needs a working JDK; javac reports no Java runtime"
  javac -d "$OUTDIR" "$SOURCE" || die "javac failed on $SOURCE"
  CP="$OUTDIR:$MODELS"
fi

# --no-lazy-methods is load-bearing: lazy loading is jbmc's default and omits
# method bodies the program reaches (§2.2).
jbmc "$CLASS" -cp "$CP" --no-lazy-methods --show-symbol-table --json-ui \
  > "$OUTDIR/symtab.json" 2> "$OUTDIR/jbmc.log" \
  || die "jbmc failed; see $OUTDIR/jbmc.log"

python3 - "$OUTDIR/symtab.json" "$OUTDIR/st.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    doc = json.load(f)
tables = [e for e in doc if "symbolTable" in e]
if not tables:
    sys.exit("no symbolTable element in jbmc --json-ui output")
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(tables[0], f)
print(f"symbols: {len(tables[0]['symbolTable'])}")
PY

symtab2gb "$OUTDIR/st.json" --out "$OUTDIR/model.goto" \
  || die "symtab2gb failed"

set +e
"$ESBMC" --binary "$OUTDIR/model.goto" "${ESBMC_ARGS[@]}" \
  > "$OUTDIR/esbmc.log" 2>&1
ESBMC_STATUS=$?
set -e

# symtab2gb has no --version; cbmc ships alongside it from the same build.
{
  echo "class:        $CLASS"
  [ -n "$SOURCE" ] && echo "source:       $SOURCE"
  echo "models-jar:   $MODELS"
  echo "jbmc:         $(jbmc --version 2>&1 | head -1)"
  echo "cbmc:         $(cbmc --version 2>&1 | head -1)"
  echo "esbmc:        $("$ESBMC" --version 2>&1 | head -1)"
  echo "esbmc-src:    $(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse \
                          --short HEAD 2>/dev/null || echo unknown)"
  echo "javac:        $(javac_version)"
  echo "goto-header:  $(od -An -tx1 -N4 "$OUTDIR/model.goto" | tr -d ' ')"
  echo "esbmc-status: $ESBMC_STATUS"
  echo "esbmc-tail:   $(tail -3 "$OUTDIR/esbmc.log" | tr '\n' ' ')"
} > "$OUTDIR/manifest.txt"

cat "$OUTDIR/manifest.txt"
exit "$ESBMC_STATUS"
