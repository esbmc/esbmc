# shellcheck shell=bash
# Shared helpers for the IREP2 goto-program migration differential harness.
# Sourced by capture_goto_baseline.sh and diff_goto_baseline.sh.
# See README.md for the rationale (tracking issue: esbmc/esbmc#4715).

# Resolve the repository root from this script's location.
irep2_repo_root() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  printf '%s\n' "$here"
}

# Absolutise a path that contains a slash (so it survives a later `cd`).
# A bare command name (no slash) is left untouched to be resolved via PATH.
irep2_abspath() {
  case "$1" in
    */*) printf '%s\n' "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")" ;;
    *)   printf '%s\n' "$1" ;;
  esac
}

# Canonicalise `esbmc --goto-functions-only` output read from stdin.
# Strips environment-specific and program-global noise while preserving
# instruction kinds, operands, guards, goto targets and file/line locations.
# $1 = repository root (collapsed to the literal token REPO_ROOT).
irep2_canon() {
  local repo="$1"
  sed \
    -e '/^ESBMC version /d' \
    -e '/^GOTO program creation time:/d' \
    -e '/^GOTO program processing time:/d' \
    -e '/ time: [0-9.]*s$/d' \
    -e 's|^\([[:space:]]*//\) [0-9][0-9]* file |\1 file |' \
    -e "s|${repo}|REPO_ROOT|g" \
    -e 's|/var/folders/[^/ ]*/[^/ ]*/T/esbmc[._-][0-9A-Za-z-]*|HDR|g' \
    -e 's|/tmp/esbmc[._-][0-9A-Za-z-]*|HDR|g'
}

# Produce the canonical goto dump for one test.desc into stdout. Returns 1 if
# the descriptor is unusable (no source / missing file).
# $1 = esbmc binary  $2 = path to a test.desc  $3 = repo root
#
# Argument construction mirrors regression/testing_tool.py exactly
# (generate_run_argument_list): line 3 is word-split, each token resolved to
# <test_dir>/<token> when that names an existing file else passed verbatim, then
# the source is appended. We add --goto-functions-only so esbmc stops after
# building the goto program. Goto-affecting flags (--data-races-check,
# --k-induction, --no-pointer-check, ...) are thus honoured exactly as the
# real suite passes them.
irep2_goto_dump() {
  local esbmc="$1" desc="$2" repo="$3"
  local dir src args tok
  dir="$(dirname "$desc")"
  src="$(sed -n '2p' "$desc" | tr -d '[:space:]')"
  args="$(sed -n '3p' "$desc")"
  [ -n "$src" ] && [ -f "$dir/$src" ] || return 1

  local argv=()
  for tok in $args; do
    if [ -f "$dir/$tok" ]; then argv+=("$dir/$tok"); else argv+=("$tok"); fi
  done

  # esbmc's exit status is intentionally ignored: --goto-functions-only can exit
  # non-zero on some inputs, but the (deterministic) dumped text is what we diff.
  ( "$esbmc" "${argv[@]}" --goto-functions-only "$dir/$src" 2>&1 || true ) \
    | irep2_canon "$repo"
}

# Remove per-run header temp dirs (CLAUDE.md hygiene rule).
irep2_cleanup_tmp() {
  rm -rf /tmp/esbmc-headers-* 2>/dev/null || true
}
