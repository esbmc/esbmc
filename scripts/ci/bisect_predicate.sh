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
output=$(ctest -R "$pattern" --output-on-failure 2>&1) && rc=0 || rc=$?
printf '%s\n' "$output"

# ctest exits 0 when its filter matches nothing, and says so on stderr, not
# stdout: a commit predating the test would otherwise read as "good" and
# bisect would blame whichever commit added the test.
case "$output" in
  *"No tests were found"*) echo "$BISECT_TEST does not exist at this commit" >&2; exit 125 ;;
esac

exit "$rc"
