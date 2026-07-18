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
    --class)  [ $# -ge 2 ] || die "--class needs a value"; CLASS="$2"; shift 2 ;;
    --source) [ $# -ge 2 ] || die "--source needs a value"; SOURCE="$2"; shift 2 ;;
    --outdir) [ $# -ge 2 ] || die "--outdir needs a value"; OUTDIR="$2"; shift 2 ;;
    --)       shift; ESBMC_ARGS=("$@"); break ;;
    -h|--help) usage ;;
    *) die "unknown argument '$1' (try --help)" ;;
  esac
done

[ -n "$CLASS" ] || usage

for tool in jbmc symtab2gb python3; do
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

# Start from an empty directory: a run that dies partway would otherwise leave
# the previous run's manifest and model.goto in place, and a reader collecting
# artefacts afterwards would get a stale, plausible-looking record of a
# different class. Stale .class files would also linger on the classpath below.
rm -rf "$OUTDIR"
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

SYMBOLS=$(python3 - "$OUTDIR/symtab.json" "$OUTDIR/st.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    doc = json.load(f)
tables = [e for e in doc if "symbolTable" in e]
if not tables:
    sys.exit("no symbolTable element in jbmc --json-ui output")
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(tables[0], f)
print(len(tables[0]["symbolTable"]))
PY
)

# Model size is the direct measurement of the completeness risk in §2.2 (16463
# symbols with --no-lazy-methods against core-models.jar, 1548 without: a
# truncated model yields a clean SUCCESSFUL, not an error). Always record it.
# MIN_SYMBOLS enforces a floor when the caller knows what to expect; there is
# no defensible universal value, so it is opt-in rather than guessed.
if [ -n "${MIN_SYMBOLS:-}" ] && [ "$SYMBOLS" -lt "$MIN_SYMBOLS" ]; then
  die "symbol table has $SYMBOLS symbols, below MIN_SYMBOLS=$MIN_SYMBOLS; \
the model is probably truncated and any verdict from it is meaningless"
fi

symtab2gb "$OUTDIR/st.json" --out "$OUTDIR/model.goto" \
  || die "symtab2gb failed"

# ${A[@]+"${A[@]}"} rather than "${A[@]}": bash 3.2 (stock on macOS) treats an
# empty array as unset under `set -u` and aborts, which would fire on the
# default invocation after all the expensive work is already done.
set +e
"$ESBMC" --binary "$OUTDIR/model.goto" ${ESBMC_ARGS[@]+"${ESBMC_ARGS[@]}"} \
  > "$OUTDIR/esbmc.log" 2>&1
ESBMC_STATUS=$?
set -e

# symtab2gb has no --version; cbmc ships alongside it from the same build.
{
  echo "class:        $CLASS"
  [ -n "$SOURCE" ] && echo "source:       $SOURCE"
  echo "models-jar:   $MODELS"
  echo "jbmc:         $(jbmc --version 2>&1 | head -1)"
  echo "cbmc:         $(cbmc --version 2>&1 | head -1 || echo 'not present')"
  echo "esbmc:        $("$ESBMC" --version 2>&1 | head -1)"
  echo "esbmc-path:   $(command -v "$ESBMC")"
  echo "javac:        $(javac_version)"
  echo "symbols:      $SYMBOLS"
  echo "goto-header:  $(od -An -tx1 -N4 "$OUTDIR/model.goto" | tr -d ' ')"
  echo "esbmc-status: $ESBMC_STATUS"
  echo "esbmc-tail:   $(tail -3 "$OUTDIR/esbmc.log" | tr '\n' ' ')"
} > "$OUTDIR/manifest.txt"

cat "$OUTDIR/manifest.txt"
exit "$ESBMC_STATUS"
