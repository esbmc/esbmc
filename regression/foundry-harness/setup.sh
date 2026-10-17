#!/usr/bin/env bash
# Reproduce the frozen Foundry environment.
#
# `lib/` is gitignored rather than vendored or added as a submodule: vendoring
# would drop ~1000 files of forge-std into this repository, and a submodule
# changes .gitmodules for everyone. The pin lives here instead, as an exact tag
# AND an exact commit, and the script verifies the commit it actually got --
# a tag can be moved, a commit cannot.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORGE_STD_TAG="v1.16.2"
FORGE_STD_SHA="bf647bd6046f2f7da30d0c2bf435e5c76a780c1b"

if [ -d "$HERE/lib/forge-std/.git" ]; then
  echo "forge-std already present at $HERE/lib/forge-std"
else
  git clone --depth 1 --branch "$FORGE_STD_TAG" \
    https://github.com/foundry-rs/forge-std "$HERE/lib/forge-std"
fi

GOT="$(git -C "$HERE/lib/forge-std" rev-parse HEAD)"
if [ "$GOT" != "$FORGE_STD_SHA" ]; then
  echo "PIN MISMATCH: expected $FORGE_STD_SHA, got $GOT" >&2
  echo "The tag $FORGE_STD_TAG no longer points where it did. Do not proceed:" >&2
  echo "numbers measured under a different forge-std are not comparable." >&2
  exit 1
fi
echo "forge-std pinned at $FORGE_STD_TAG ($GOT)"

echo "forge:  $(forge --version | head -c 200)"
echo "solc :  $(solc --version | tail -n 1)"
