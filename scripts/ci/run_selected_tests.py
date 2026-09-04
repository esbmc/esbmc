#!/usr/bin/env python3
"""Run exactly the tests select_tests.py chose, in one ctest invocation.

ctest can only be narrowed by name regex (``-R``) or by test number (``-I``).
A name alternation over a few thousand tests is ~170 kB, past the 128 kB Linux
caps on a single argument (MAX_ARG_STRLEN), so the selection is resolved to test
numbers instead: ``ctest --show-only=json-v1`` lists tests in the same order
ctest numbers them, and the resulting index list stays under 90 kB even if every
test in the suite is selected.

Numbers are resolved against the build directory being run, so they cannot go
stale the way a checked-in index list would.
"""

import argparse
import json
import os
import subprocess
import sys

# The sibling import below has to follow the sys.path setup.
# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_tests import read_lines  # noqa: E402  (sibling module, not a package)

# MAX_ARG_STRLEN is 128 kB; stop short of it with room for the rest of argv.
MAX_ARG_BYTES = 120_000


def ctest_test_names(build_dir):
    """List the build's tests in ctest numbering order (test #N is entry N-1)."""
    out = subprocess.run(["ctest", "--show-only=json-v1"],
                         cwd=build_dir,
                         check=True,
                         capture_output=True,
                         text=True).stdout
    return [t["name"] for t in json.loads(out)["tests"]]


def resolve(selected, names):
    """Map test names to 1-based ctest numbers. Returns ``(numbers, missing)``."""
    index = {name: i + 1 for i, name in enumerate(names)}
    numbers = sorted(index[n] for n in selected if n in index)
    return numbers, sorted(n for n in selected if n not in index)


def main(argv=None):
    """Resolve the selection to ctest numbers and run it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-dir", default="build", help="configured build directory")
    parser.add_argument("--tests", required=True, help="selected test list from select_tests.py")
    parser.epilog = "Any other argument is forwarded to ctest verbatim."
    # parse_known_args, not parse_args: ctest's own flags (-j, -N,
    # --output-on-failure) must pass straight through.
    args, ctest_args = parser.parse_known_args(argv)

    selected = read_lines(args.tests)
    names = ctest_test_names(args.build_dir)
    numbers, missing = resolve(selected, names)

    if missing:
        # Expected whenever the run's build enables fewer suites than the one
        # that produced the selection (no bitwuzla, no Solidity frontend, ...).
        print(f"note: {len(missing)} selected tests are not in this build, e.g. {missing[:3]}",
              file=sys.stderr)
    if not numbers:
        print("error: none of the selected tests exist in this build", file=sys.stderr)
        return 1

    spec = "0,0,0," + ",".join(str(n) for n in numbers)
    if len(spec) > MAX_ARG_BYTES:
        print(f"error: {len(numbers)} tests exceed the ctest argument limit; lower the budget",
              file=sys.stderr)
        return 1

    cmd = ["ctest", "-I", spec] + ctest_args
    print(f"running {len(numbers)} of {len(names)} tests", file=sys.stderr)
    return subprocess.run(cmd, cwd=args.build_dir, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
