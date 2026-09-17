#!/bin/bash
# Full-corpus verdict census of every Python regression row under
# --python-irep2-adjust-only, against the default path. One TSV line per row:
#   STATUS<TAB>dir<TAB>off<TAB>on
# STATUS is AGREE, DIVERGE, CRASH (on-side SIGSEGV/uncaught exception) or SKIP.
# Resumable: rows already present in $OUT are not re-run.
E=/home/rafaelsa/esbmc/build/src/esbmc/esbmc
OUT=${1:?usage: cpp_full_census.sh <out.tsv> [jobs]}
JOBS=${2:-4}
cd /home/rafaelsa/esbmc || exit 1
touch "$OUT"

run_row() {
  local desc="$1" out="$2" E="$3"
  local d; d=$(dirname "$desc")
  grep -qP "^[A-Z]+\t\Q$d\E\t" "$out" 2>/dev/null && return
  local kind src flags
  kind=$(sed -n '1p' "$desc"); src=$(sed -n '2p' "$desc"); flags=$(sed -n '3p' "$desc")
  case "$kind" in KNOWNBUG|FUTURE) printf 'SKIP\t%s\tknownbug\t-\n' "$d" >> "$out"; return;; esac
  case "$flags" in *irep2*) printf 'SKIP\t%s\tpins-flag\t-\n' "$d" >> "$out"; return;; esac
  [ -f "$d/$src" ] || { printf 'SKIP\t%s\tno-source\t-\n' "$d" >> "$out"; return; }
  # The flags line is a list of words, so it is split deliberately; an
  # array does that without exposing the words to globbing too.
  local -a flag_words
  read -r -a flag_words <<<"$flags"
  local off on voff von
  off=$( (cd "$d" && timeout 45 "$E" "$src" "${flag_words[@]}" 2>&1) )
  on=$(  (cd "$d" && timeout 45 "$E" "$src" "${flag_words[@]}" --python-irep2-adjust-only 2>&1) )
  voff=$(grep -oE '^VERIFICATION (SUCCESSFUL|FAILED|UNKNOWN)' <<<"$off" | tail -1)
  von=$( grep -oE '^VERIFICATION (SUCCESSFUL|FAILED|UNKNOWN)' <<<"$on"  | tail -1)
  [ -z "$voff" ] && voff=none
  [ -z "$von" ] && von=none
  if grep -qE '^ESBMC caught SIGSEGV|^ERROR: uncaught exception|tuple field out of range' <<<"$on"; then
    printf 'CRASH\t%s\t%s\t%s\n' "$d" "$voff" "$von" >> "$out"
  elif [ "$voff" = "$von" ]; then
    printf 'AGREE\t%s\t%s\t%s\n' "$d" "$voff" "$von" >> "$out"
  else
    printf 'DIVERGE\t%s\t%s\t%s\n' "$d" "$voff" "$von" >> "$out"
  fi
}
export -f run_row
find regression/python regression/python-contracts \
     regression/python-coverage \
     -name test.desc | sort \
  | xargs -P "$JOBS" -I{} bash -c 'run_row "{}" "'"$OUT"'" "'"$E"'"'
echo "--- census complete: $(wc -l < "$OUT") rows"
