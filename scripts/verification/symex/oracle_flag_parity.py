#!/usr/bin/env python3
"""Tier-C metamorphic oracle: two flag-sets must agree on every verdict.

Implements H-C1, H-C2, H-C3 and H-C5 of docs/roadmap/goto-symex-verification-
plan.md (§7.4). Each of those is the same relation with different flags, so they
share one driver rather than four near-identical scripts:

    H-C1  slice parity       --b=--no-slice
    H-C2  simplify parity    --b=--no-simplify
    H-C3  solver parity      --a=--bitwuzla --b=--z3
    H-C5  interval parity    --b=--no-interval-symex-guard

Flag values must use `--b=...` form: argparse reads a bare `--b --no-slice` as a
missing value followed by an unknown option.

These are pure verdict comparisons: no modelling and no assumptions, so a
divergence is a real defect in one of the two configurations. What the oracle
cannot do is say which one -- that is triage, and §11.3 requires every
divergence to reach a filed issue or a reviewed waiver before it may be
baselined.

The argument list per test comes from regression/testing_tool.py's TestCase, so
a sweep runs each input exactly as ctest does (same flag order, same
--timeout/--memlimit stripping); reimplementing that parsing is how a sweep ends
up reporting divergences that are really invocation differences.

Usage:
    oracle_flag_parity.py --esbmc build/src/esbmc/esbmc --b=--no-slice
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

REGRESSION = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "regression"
)
sys.path.insert(0, os.path.abspath(REGRESSION))

# pylint: disable=import-error,wrong-import-position
from testing_tool import TestCase  # type: ignore

VERDICT = re.compile(r"^VERIFICATION (SUCCESSFUL|FAILED|UNKNOWN)$", re.MULTILINE)

# Not a verdict: the run never reached one, so the pair says nothing about
# parity. Counted and reported rather than dropped -- a sweep that silently
# skips its hard cases reads as "1430 inputs agree" when it is not.
NO_VERDICT = "no-verdict"
TIMEOUT = "timeout"


def verdict_of(esbmc, args, cwd, timeout):
    try:
        proc = subprocess.run(
            [esbmc] + args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,  # a FAILED verdict exits non-zero and is a result
        )
    except subprocess.TimeoutExpired:
        return TIMEOUT
    found = VERDICT.findall(proc.stdout.decode("utf-8", "replace"))
    # Under --multi-property ESBMC prints one line per property; the run's
    # verdict is FAILED if any property failed.
    if not found:
        return NO_VERDICT
    return "FAILED" if "FAILED" in found else found[-1]


def collect(suite, modes):
    # Absolute: TestCase resolves the input file against test_dir, and each pair
    # runs in a scratch cwd where a repo-relative path would not exist.
    suite = os.path.abspath(suite)
    tests = []
    for entry in sorted(os.listdir(suite)):
        directory = os.path.join(suite, entry)
        if not os.path.isfile(os.path.join(directory, "test.desc")):
            continue
        case = TestCase(directory, entry)
        if case.test_mode in modes:
            tests.append(case)
    return tests


def run_pair(case, esbmc, flags_a, flags_b, timeout):
    """Both flag-sets on one input, in a scratch cwd so output files cannot
    collide between the two runs or between concurrent tests."""
    base = case.generate_run_argument_list(esbmc)[1:]
    work = tempfile.mkdtemp(prefix="oracle-parity-")
    try:
        a = verdict_of(esbmc, flags_a + base, work, timeout)
        b = verdict_of(esbmc, flags_b + base, work, timeout)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return case.name, a, b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esbmc", default="build/src/esbmc/esbmc")
    parser.add_argument("--suite", default="regression/esbmc")
    parser.add_argument("--a", default="", help="extra flags for the baseline run")
    parser.add_argument("--b", required=True, help="extra flags for the variant run")
    parser.add_argument("--modes", default="CORE")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--baseline",
        help="file of test names already triaged as diverging (one per line, "
        "'#' comments allowed). New divergences fail; a baselined test that "
        "stops diverging is reported so the file cannot rot unnoticed.",
    )
    args = parser.parse_args()

    esbmc = os.path.abspath(args.esbmc)
    if not os.access(esbmc, os.X_OK):
        parser.error(f"not executable: {esbmc}")

    flags_a = args.a.split()
    flags_b = args.b.split()
    modes = args.modes.split(",")
    tests = collect(args.suite, modes)

    # A test already naming one of the flags would compare a configuration
    # against itself, or against one the author deliberately pinned.
    named = set(flags_a + flags_b)
    skipped = [t for t in tests if named & set(t.test_args.split())]
    tests = [t for t in tests if t not in skipped]
    if args.limit:
        tests = tests[: args.limit]

    baseline = set()
    if args.baseline:
        with open(args.baseline, "r", encoding="utf-8") as handle:
            for line in handle:
                name = line.split("#", 1)[0].strip()
                if name:
                    baseline.add(name)

    print(f"{len(tests)} tests, modes={modes}, A={flags_a or ['(none)']} B={flags_b}")

    diverged, inconclusive, agreed = [], [], 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(run_pair, t, esbmc, flags_a, flags_b, args.timeout)
            for t in tests
        ]
        for done, future in enumerate(futures, 1):
            name, a, b = future.result()
            if a in (TIMEOUT, NO_VERDICT) or b in (TIMEOUT, NO_VERDICT):
                inconclusive.append((name, a, b))
            elif a != b:
                diverged.append((name, a, b))
                print(f"  DIVERGE {name}: A={a} B={b}", flush=True)
            else:
                agreed += 1
            if done % 100 == 0:
                print(f"  ... {done}/{len(tests)}", flush=True)

    diverged_names = {name for name, _, _ in diverged}
    new_divergences = sorted(diverged_names - baseline)
    # A baselined test that now agrees means the underlying defect was fixed;
    # keeping the entry would mask a later regression of the same test.
    stale_baseline = sorted(baseline - diverged_names)

    print(f"\nagreed       {agreed}")
    print(f"diverged     {len(diverged)}")
    print(f"inconclusive {len(inconclusive)}  (no verdict or timeout in one leg)")
    print(f"skipped      {len(skipped)}  (test.desc already names a compared flag)")
    for name, a, b in diverged:
        print(f"DIVERGE {name}: A={a} B={b}")
    for name in sorted(t.name for t in skipped):
        print(f"SKIP {name}")
    for name, a, b in sorted(inconclusive):
        print(f"INCONCLUSIVE {name}: A={a} B={b}")

    if baseline:
        print(f"\nbaselined    {len(diverged_names & baseline)} of {len(baseline)}")
        for name in stale_baseline:
            print(f"STALE-BASELINE {name}: listed as diverging but agrees now")
        for name in new_divergences:
            print(f"NEW-DIVERGENCE {name}")

    return 1 if new_divergences else 0


if __name__ == "__main__":
    sys.exit(main())
