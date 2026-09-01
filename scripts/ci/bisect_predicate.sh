#!/bin/sh
# git-bisect predicate for the nightly regression hunt (esbmc/esbmc#6735).
#
# Exit 0 when the commit is good and non-zero when it is bad. Exit 125 means
# "cannot judge": git bisect skips such a commit, so one that fails to build is
# stepped over rather than blamed for a regression it may not have caused.
#
# BISECT_TEST names the single ctest test being bisected.
set -e

[ -n "$BISECT_TEST" ] || { echo "BISECT_TEST is not set" >&2; exit 125; }

./scripts/build.sh -b DebugOpt -e ON >/dev/null 2>&1 || exit 125

# The test name is a literal, not a pattern: ctest -R takes a regex, and ESBMC
# test names contain characters that are regex metacharacters.
pattern=$(python3 -c 'import os, re; print("^" + re.escape(os.environ["BISECT_TEST"]) + "$")')

cd build
ctest -R "$pattern" --output-on-failure
