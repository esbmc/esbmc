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

That holds only while both legs decide the same question, which is why tests
selecting an approximate arithmetic encoding are skipped. `--ir` reasons over
unbounded integers/reals and `--fixedbv` models floats as fixed-point, so
neither decides the C program: a verdict can turn on how much the *simplifier*
folded in exact C semantics before encoding, and `verdict(default) !=
verdict(--no-simplify)` is then the encoding's approximation showing through
rather than a defect. Each such case was confirmed by re-running without the
encoding flag, where the two legs agree again (§15 M9 R16).

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
import time
from collections import namedtuple
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

# Flags selecting an encoding that does not decide the C program: `--ir` uses
# unbounded integers/reals, `--fixedbv` fixed-point instead of IEEE floats. A
# verdict under one of these is a property of the abstraction, so comparing it
# across a flag that changes constant folding proves nothing either way.
APPROXIMATE_ENCODINGS = {"--ir", "--ir-ieee", "--fixedbv"}

# The binary and the two flag-sets are one thing -- the comparison being run --
# and every function here needs all three or none of them.
Legs = namedtuple("Legs", "esbmc flags_a flags_b")

# What the sweep decided, per §15 M9 (H-C2 residue): `no_verdict` and
# `timed_out` are separate because one is a property of the input and the other
# of the run.
Outcome = namedtuple("Outcome", "agreed diverged no_verdict timed_out")


def run_pair(case, legs, timeout):
    """Both flag-sets on one input, in a scratch cwd so output files cannot
    collide between the two runs or between concurrent tests."""
    base = case.generate_run_argument_list(legs.esbmc)[1:]
    work = scratch_dir("oracle-parity-")
    try:
        a = verdict_of(capture(legs.esbmc, legs.flags_a + base, work, timeout))
        b = verdict_of(capture(legs.esbmc, legs.flags_b + base, work, timeout))
    finally:
        drop_scratch(work)
    return case, a, b


def run_pass(cases, legs, timeout, jobs, announce_diverged):
    """Every case through both legs, `jobs` at a time.

    A whole-corpus sweep runs for an hour or more, so a divergence is announced
    as it lands rather than only in the closing report -- but only when no retry
    pass follows, since a first-pass timeout can still become a verdict and turn
    an announced divergence into a retraction.
    """
    results = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_pair, c, legs, timeout) for c in cases]
        for done, future in enumerate(futures, 1):
            case, a, b = future.result()
            results.append((case, a, b))
            if announce_diverged and a != b and TIMEOUT not in (a, b):
                print(f"  DIVERGE {case.name}: A={a} B={b}", flush=True)
            if done % 100 == 0:
                print(f"  ... {done}/{len(cases)}", flush=True)
    return results


def retry_serially(cases, legs, timeout, budget):
    """Re-run each case on its own at a larger bound, until the budget is spent.

    Serial because the point is to remove self-contention, and budgeted because
    the worst case -- every case using the full bound in both legs -- runs for
    hours. Returns (results, cases the budget did not reach).
    """
    deadline = time.monotonic() + budget
    results = []
    for index, case in enumerate(cases):
        if time.monotonic() >= deadline:
            return results, cases[index:]
        results.append(run_pair(case, legs, timeout))
    return results, []


def sweep(tests, legs, args):
    """Both flag-sets over every test, then a second pass over what timed out.

    A timeout at `--jobs` concurrency need not be a property of the input: the
    first pass runs that many ESBMC pairs at once, so an input anywhere near the
    bound can lose to load, and a count that moves with `uptime` is not a
    measurement. The second pass re-runs only that residue, alone and at a
    larger bound, which separates the two -- and prints how many it settled, so
    a budget that is buying nothing is visible rather than assumed.
    """
    results = run_pass(
        tests, legs, args.timeout, args.jobs, not args.retry_timeout)
    timed_out = [c for c, a, b in results if TIMEOUT in (a, b)]
    if not (timed_out and args.retry_timeout):
        return results

    print(
        f"\nretrying {len(timed_out)} timed-out tests serially "
        f"at {args.retry_timeout}s",
        flush=True,
    )
    retried, unreached = retry_serially(
        timed_out, legs, args.retry_timeout, args.retry_budget
    )
    settled = {c.name: (a, b) for c, a, b in retried}
    recovered = sum(1 for a, b in settled.values() if TIMEOUT not in (a, b))
    print(f"  settled {recovered} of {len(retried)} retried", flush=True)
    if unreached:
        print(f"  budget spent; {len(unreached)} not retried", flush=True)
    return [(c, *settled.get(c.name, (a, b))) for c, a, b in results]


def classify(results):
    """Split the sweep four ways.

    `no-verdict` and `timeout` were one "inconclusive" count until §15 M9
    (H-C2 residue), which is one number for two unlike things: reaching no
    verdict is a property of the input and reproduces on any machine, whereas a
    timeout is a property of the run. Reported together, a stable exclusion
    cannot be told from a load artefact.
    """
    agreed, diverged, no_verdict, timed_out = 0, [], [], []
    for case, a, b in results:
        row = (case.name, a, b)
        if TIMEOUT in (a, b):
            timed_out.append(row)
        elif NO_VERDICT in (a, b):
            no_verdict.append(row)
        elif a != b:
            diverged.append(row)
        else:
            agreed += 1
    return Outcome(agreed, diverged, no_verdict, timed_out)


def report(outcome, skipped, abstract):
    """Counts first, then every excluded and diverging test by name -- a sweep
    that hides what it dropped reads as broader coverage than it had."""
    agreed, diverged, no_verdict, timed_out = outcome
    print(f"\nagreed       {agreed}")
    print(f"diverged     {len(diverged)}")
    print(f"no-verdict   {len(no_verdict)}  (a leg reached no verdict on this input)")
    print(f"timeout      {len(timed_out)}  (still over the bound after the retry)")
    print(f"skipped      {len(skipped)}  (test.desc already names a compared flag)")
    print(f"abstract     {len(abstract)}  (test.desc selects an approximate encoding)")
    for name, a, b in diverged:
        print(f"DIVERGE {name}: A={a} B={b}")
    for name in sorted(t.name for t in skipped):
        print(f"SKIP {name}")
    for name in sorted(t.name for t in abstract):
        print(f"ABSTRACT {name}")
    for name, a, b in sorted(no_verdict):
        print(f"NO-VERDICT {name}: A={a} B={b}")
    for name, a, b in sorted(timed_out):
        print(f"TIMEOUT {name}: A={a} B={b}")


def main():
    """Exit non-zero on a divergence the baseline does not already carry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esbmc", default="build/src/esbmc/esbmc")
    parser.add_argument("--suite", default="regression/esbmc")
    parser.add_argument("--a", default="", help="extra flags for the baseline run")
    parser.add_argument("--b", required=True, help="extra flags for the variant run")
    parser.add_argument("--modes", default="CORE")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument(
        "--retry-timeout",
        type=int,
        default=120,
        help="bound for the serial second pass over tests that timed out under "
        "--jobs concurrency; 0 disables the pass",
    )
    parser.add_argument(
        "--retry-budget",
        type=int,
        default=900,
        help="wall-clock seconds the second pass may spend; tests it does not "
        "reach stay reported as timeouts",
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--baseline",
        help="file of test names already triaged as diverging (one per line, "
        "'#' comments allowed). New divergences fail; a baselined test that "
        "stops diverging is reported so the file cannot rot unnoticed.",
    )
    args = parser.parse_args()

    legs = Legs(esbmc_path(parser, args.esbmc), args.a.split(), args.b.split())
    tests = collect_tests(args.suite, args.modes.split(","))

    # A test already naming one of the flags would compare a configuration
    # against itself, or against one the author deliberately pinned.
    named = set(legs.flags_a + legs.flags_b)
    skipped = [t for t in tests if named & set(t.test_args.split())]
    abstract = [
        t
        for t in tests
        if t not in skipped and APPROXIMATE_ENCODINGS & set(t.test_args.split())
    ]
    tests = [t for t in tests if t not in skipped and t not in abstract]
    if args.limit:
        tests = tests[: args.limit]

    baseline = load_baseline(args.baseline)

    print(
        f"{len(tests)} tests, modes={args.modes}, "
        f"A={legs.flags_a or ['(none)']} B={legs.flags_b}"
    )

    outcome = classify(sweep(tests, legs, args))
    report(outcome, skipped, abstract)
    diverged_names = {name for name, _, _ in outcome.diverged}
    return 1 if report_baseline(baseline, diverged_names) else 0


if __name__ == "__main__":
    sys.exit(main())
