#!/usr/bin/env bash
# Push a real generated test through the environment and report the verdict.
#
# The point is that "forge-std is installed" is not the claim worth making. The
# claim worth making is that something the generator actually emitted compiles
# and runs here, so this copies from a regression case rather than keeping its
# own copy of a contract -- a second copy would drift from the first and the
# smoke test would start certifying the wrong file.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE="${1:-foundry_covgen_scalar_fail}"
SRC="$HERE/../esbmc-solidity/$CASE"

if [ ! -d "$SRC" ]; then
  echo "no such regression case: $SRC" >&2
  exit 1
fi

shopt -s nullglob
TESTS=("$SRC"/*.cov.t.sol)
if [ ${#TESTS[@]} -eq 0 ]; then
  echo "case $CASE has no generated *.cov.t.sol to run." >&2
  echo "Generated tests are produced by a run, not checked in for every case." >&2
  exit 1
fi

rm -f "$HERE"/test/*.sol
cp "$SRC"/contract.sol "$HERE/test/contract.sol"
for t in "${TESTS[@]}"; do cp "$t" "$HERE/test/$(basename "$t")"; done

cd "$HERE"
forge test -vv
