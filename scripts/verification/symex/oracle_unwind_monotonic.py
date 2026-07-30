#!/usr/bin/env python3
"""Tier-C oracle: a real counterexample must survive a larger unwind bound.

H-C6 of docs/roadmap/goto-symex-verification-plan.md (§7.4). Raising --unwind can
only add behaviours, so a genuine property violation found at bound k must still
be found at every k' > k. Losing it means symex dropped a counterexample as the
bound grew -- the missed-bug direction, and unlike the flag-parity oracles this
one needs no reference configuration: monotonicity is a soundness relation the
tool must satisfy against itself.

One distinction carries the whole oracle. With unwinding assertions enabled (the
default) a FAILED at a small bound is often the *unwinding assertion* itself --
"loop not fully unwound" -- which correctly disappears once the bound covers the
loop. Treating that as a lost counterexample would report a divergence on every
loop-bearing test in the corpus. So verdicts are classified by the violated
property, not by the FAILED line, and only a non-unwinding violation is held to
the relation.

Usage:
    oracle_unwind_monotonic.py --esbmc build/src/esbmc/esbmc --bounds 1,2,4,8
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
UNWIND_CLAIM = re.compile(r"unwinding assertion loop", re.IGNORECASE)

# A bound whose run said nothing about the property: excluded from the relation
# and reported, never silently treated as agreement.
TIMEOUT = "timeout"
NO_VERDICT = "no-verdict"
# FAILED on the unwinding assertion: a bound artefact, not a counterexample.
FAILED_UNWIND = "failed-unwind"
FAILED_REAL = "failed-real"

# Options that pin the bound themselves; overriding them would not be the test
# the author wrote.
BOUND_OPTIONS = (
    "--unwind",
    "--unwindset",
    "--unwindsetname",
    "--k-induction",
    "--incremental-bmc",
    "--falsification",
    "--termination",
)

LOOP_HINT = re.compile(r"\bwhile\b|\bfor\s*\(|\bgoto\b|\bdo\b")


def classify(esbmc, args, cwd, timeout):
    try:
        proc = subprocess.run(
            [esbmc] + args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return TIMEOUT
    out = proc.stdout.decode("utf-8", "replace")
    found = VERDICT.findall(out)
    if not found:
        return NO_VERDICT
    if "FAILED" not in found:
        return found[-1]
    return FAILED_UNWIND if UNWIND_CLAIM.search(out) else FAILED_REAL


def loop_bearing(case):
    """Cheap over-approximation: a test with no loop keyword cannot lose a
    counterexample to a bound, and including one costs a run per bound."""
    if not case.test_file:
        return False
    path = os.path.join(case.test_dir, case.test_file)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return bool(LOOP_HINT.search(handle.read()))
    except OSError:
        return False


def collect(suite, modes):
    suite = os.path.abspath(suite)
    tests = []
    for entry in sorted(os.listdir(suite)):
        directory = os.path.join(suite, entry)
        if not os.path.isfile(os.path.join(directory, "test.desc")):
            continue
        case = TestCase(directory, entry)
        if case.test_mode not in modes:
            continue
        if any(opt in case.test_args for opt in BOUND_OPTIONS):
            continue
        if not loop_bearing(case):
            continue
        tests.append(case)
    return tests


def run_ladder(case, esbmc, bounds, timeout):
    base = case.generate_run_argument_list(esbmc)[1:]
    work = tempfile.mkdtemp(prefix="oracle-unwind-")
    try:
        return case.name, [
            classify(esbmc, ["--unwind", str(k)] + base, work, timeout)
            for k in bounds
        ]
    finally:
        shutil.rmtree(work, ignore_errors=True)


def first_violation(bounds, verdicts):
    """The relation: once FAILED_REAL at bound k, every larger conclusive bound
    must also be FAILED_REAL."""
    for i, low in enumerate(verdicts):
        if low != FAILED_REAL:
            continue
        for j in range(i + 1, len(verdicts)):
            if verdicts[j] in (TIMEOUT, NO_VERDICT):
                continue
            if verdicts[j] != FAILED_REAL:
                return bounds[i], bounds[j], verdicts[j]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esbmc", default="build/src/esbmc/esbmc")
    parser.add_argument("--suite", default="regression/esbmc")
    parser.add_argument("--bounds", default="1,2,4,8")
    parser.add_argument("--modes", default="CORE")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    esbmc = os.path.abspath(args.esbmc)
    if not os.access(esbmc, os.X_OK):
        parser.error(f"not executable: {esbmc}")

    bounds = [int(b) for b in args.bounds.split(",")]
    if bounds != sorted(bounds):
        parser.error("--bounds must be ascending")

    tests = collect(args.suite, args.modes.split(","))
    if args.limit:
        tests = tests[: args.limit]
    print(f"{len(tests)} loop-bearing tests, bounds={bounds}")

    violations, exercised, inconclusive = [], 0, 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(run_ladder, t, esbmc, bounds, args.timeout) for t in tests
        ]
        for done, future in enumerate(futures, 1):
            name, verdicts = future.result()
            if any(v in (TIMEOUT, NO_VERDICT) for v in verdicts):
                inconclusive += 1
            # Only a test that produced a real counterexample somewhere can
            # exercise the relation at all.
            if FAILED_REAL in verdicts:
                exercised += 1
            bad = first_violation(bounds, verdicts)
            if bad:
                violations.append((name, bad, verdicts))
                print(
                    f"  LOST k={bad[0]} FAILED -> k={bad[1]} {bad[2]}: {name}",
                    flush=True,
                )
            if done % 50 == 0:
                print(f"  ... {done}/{len(tests)}", flush=True)

    print(f"\nexercised    {exercised}  (produced a real counterexample at some bound)")
    print(f"violations   {len(violations)}")
    print(f"inconclusive {inconclusive}  (a timeout or no verdict at some bound)")
    for name, bad, verdicts in violations:
        print(f"LOST {name}: k={bad[0]} FAILED -> k={bad[1]} {bad[2]}  ladder={verdicts}")

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
