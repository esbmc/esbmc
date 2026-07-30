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
import sys
from concurrent.futures import ThreadPoolExecutor

from oracle_common import (
    NO_VERDICT,
    TIMEOUT,
    capture,
    collect_tests,
    drop_scratch,
    esbmc_path,
    load_baseline,
    report_baseline,
    scratch_dir,
    verdict_of,
)


def run_pair(case, esbmc, flags_a, flags_b, timeout):
    """Both flag-sets on one input, in a scratch cwd so output files cannot
    collide between the two runs or between concurrent tests."""
    base = case.generate_run_argument_list(esbmc)[1:]
    work = scratch_dir("oracle-parity-")
    try:
        a = verdict_of(capture(esbmc, flags_a + base, work, timeout))
        b = verdict_of(capture(esbmc, flags_b + base, work, timeout))
    finally:
        drop_scratch(work)
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

    esbmc = esbmc_path(parser, args.esbmc)

    flags_a = args.a.split()
    flags_b = args.b.split()
    modes = args.modes.split(",")
    tests = collect_tests(args.suite, modes)

    # A test already naming one of the flags would compare a configuration
    # against itself, or against one the author deliberately pinned.
    named = set(flags_a + flags_b)
    skipped = [t for t in tests if named & set(t.test_args.split())]
    tests = [t for t in tests if t not in skipped]
    if args.limit:
        tests = tests[: args.limit]

    baseline = load_baseline(args.baseline)

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

    return 1 if report_baseline(baseline, diverged_names) else 0


if __name__ == "__main__":
    sys.exit(main())
