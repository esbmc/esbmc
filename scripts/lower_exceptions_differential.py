#!/usr/bin/env python3
"""Differential gate for ``--lower-exceptions`` (issue #5075, the P4 gate).

For every exception-bearing regression test, run ESBMC twice — once on the
default imperative exception path and once with ``--lower-exceptions`` — using
the *exact* command the regression harness would build, and compare the two
verdicts. A divergence (ON reaches a different SUCCESSFUL/FAILED verdict than
OFF, or one errors where the other does not) means the lowered path is not yet
at parity, so the script exits non-zero. This is the gate that must come up
green before the imperative path (``src/goto-symex/symex_catch.cpp``) can be
deleted and the flag flipped on by default.

Invocation mirrors ``regression/testing_tool.py``: line 1 of ``test.desc`` is
the mode, line 2 the source file, line 3 the ESBMC flags. The command is
``esbmc <resolved-flags> <source>``; the ON command appends
``--lower-exceptions``. Tests whose flags already pass ``--lower-exceptions``
(the dedicated lowered-path tests) have no OFF baseline and are skipped.

Examples:
    # Default: scan the C++ and Python suites, CORE tests only.
    scripts/lower_exceptions_differential.py --esbmc build/src/esbmc/esbmc

    # Narrow to one suite, print every test as it runs.
    scripts/lower_exceptions_differential.py -e build/src/esbmc/esbmc \\
        --root regression/esbmc-cpp/try_catch --verbose

    # Just list the selected corpus without running anything.
    scripts/lower_exceptions_differential.py --list
"""

import argparse
import concurrent.futures
import os
import re
import shlex
# A developer tool, not a service: every argv comes from the repo's own
# test.desc files plus the CLI-supplied esbmc path, and is passed as an argv
# list (never a shell string, never shell=True), so there is no shell or
# untrusted-input injection surface.
import subprocess  # nosec B404
import sys

# Substrings selecting exception-bearing tests. A test is in the corpus if any
# source file in its directory contains one of these (word-boundary matched for
# the short C++ tokens to avoid e.g. "category" matching "cat"... none here, but
# "throw"/"catch"/"try" are matched verbatim as they are unambiguous in code).
CPP_KEYWORDS = ("throw", "catch", " try", "\ttry", "dynamic_cast")
PY_KEYWORDS = ("try:", "except", "raise ", "raise\n")

VERDICT_RE = re.compile(r"VERIFICATION (SUCCESSFUL|FAILED)")

# Verdict classification returned by run_one().
SUCCESSFUL = "SUCCESSFUL"
FAILED = "FAILED"
ERROR = "ERROR"  # esbmc exited without a verdict (parse error, crash, ...)
TIMEOUT = "TIMEOUT"


def find_source_files(test_dir):
    """Source files in a test directory that could carry exception constructs."""
    out = []
    for name in os.listdir(test_dir):
        if name.endswith((".cpp", ".cc", ".cxx", ".c", ".py", ".hpp", ".h")):
            out.append(os.path.join(test_dir, name))
    return out


def is_exception_bearing(test_dir, source):
    """True if the test exercises exceptions (so the differential is meaningful)."""
    keywords = PY_KEYWORDS if source.endswith(".py") else CPP_KEYWORDS
    for path in find_source_files(test_dir):
        try:
            with open(path, encoding="utf-8", errors="replace") as fp:
                text = fp.read()
        except OSError:
            continue
        if any(kw in text for kw in keywords):
            return True
    return False


class Test:  # pylint: disable=too-few-public-methods
    """A single regression test parsed from its ``test.desc``."""

    __slots__ = ("dir", "name", "mode", "source", "args")

    def __init__(self, test_dir, mode, source, args):
        self.dir = test_dir
        self.name = os.path.relpath(test_dir)
        self.mode = mode
        self.source = source
        self.args = args

    def base_command(self, esbmc):
        """The OFF command, mirroring testing_tool.generate_run_argument_list."""
        cmd = [esbmc]
        for tok in shlex.split(self.args):
            if not tok:
                continue
            candidate = os.path.join(self.dir, tok)
            cmd.append(candidate if os.path.exists(candidate) else tok)
        cmd.append(os.path.join(self.dir, self.source))
        return cmd


def parse_test(test_dir):
    """Parse a test.desc into (Test, expected_output) or None if not runnable.

    ``expected_output`` is the regex section (line 4+), used to decide whether
    the test asserts a VERIFICATION verdict — the only case the differential can
    compare. A test that expects a frontend error (``PARSING ERROR``), an
    unsupported-feature message (``ERROR: ... not supported``), or anything else
    non-verdict never reaches or is unaffected by the post-GOTO lowering pass.
    """
    desc = os.path.join(test_dir, "test.desc")
    if not os.path.isfile(desc):
        return None
    with open(desc, encoding="utf-8", errors="replace") as fp:
        head = [fp.readline().strip() for _ in range(3)]
        expected = fp.read()
    mode, source, args = head
    if not source or not os.path.exists(os.path.join(test_dir, source)):
        return None
    return Test(test_dir, mode, source, args), expected


def collect_tests(roots, modes):
    """Walk roots for exception-bearing tests in the requested modes."""
    tests = []
    for root in roots:
        for dirpath, _subdirs, files in os.walk(root):  # noqa: B007
            if "test.desc" not in files:
                continue
            parsed = parse_test(dirpath)
            if parsed is None:
                continue
            test, expected = parsed
            if modes and test.mode not in modes:
                continue
            # The dedicated lowered-path tests have no OFF baseline.
            if "--lower-exceptions" in test.args:
                continue
            # Only verdict tests are differential-comparable.
            if not VERDICT_RE.search(expected):
                continue
            if not is_exception_bearing(dirpath, test.source):
                continue
            tests.append(test)
    tests.sort(key=lambda t: t.name)
    return tests


def run_esbmc(cmd, timeout):
    """Run one ESBMC command; return its verdict / ERROR / TIMEOUT."""
    try:
        # cmd is an argv list (no shell=True) of repo-controlled test flags and
        # the CLI-supplied esbmc binary; no shell/untrusted-input injection.
        proc = subprocess.run(
            cmd,  # nosec B603
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
            errors="replace",
            check=False)
    except subprocess.TimeoutExpired:
        return TIMEOUT
    match = VERDICT_RE.search(proc.stdout)
    return match.group(1) if match else ERROR


def run_one(test, esbmc, timeout):
    """Run a test OFF and ON; return (test, off_verdict, on_verdict)."""
    base = test.base_command(esbmc)
    off = run_esbmc(base, timeout)
    on = run_esbmc(base + ["--lower-exceptions"], timeout)
    return test, off, on


def is_divergence(off, on):
    """A divergence is a meaningful disagreement between the two paths.

    Identical verdicts agree. Two ERROR/TIMEOUT outcomes are treated as
    agreement (the lowering did not change a non-verdict outcome); a verdict on
    one side and a non-verdict on the other, or two different verdicts, is a
    divergence.
    """
    if off == on:
        return False
    non_verdict = {ERROR, TIMEOUT}
    if off in non_verdict and on in non_verdict:
        return False
    return True


def run_suite(tests, args):
    """Run every test OFF/ON in parallel; return (matched, no_baseline, divergences)."""
    divergences = []
    no_baseline = []
    matched = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, t, args.esbmc, args.timeout) for t in tests]
        for done in concurrent.futures.as_completed(futures):
            test, off, on = done.result()
            if is_divergence(off, on):
                divergences.append((test, off, on))
                tag = "DIVERGE"
            elif off in (SUCCESSFUL, FAILED) and on in (SUCCESSFUL, FAILED):
                matched += 1
                tag = "ok"
            else:
                no_baseline.append((test, off, on))
                tag = "skip"  # both errored/timed out — no usable baseline
            if args.verbose or tag == "DIVERGE":
                print(f"  [{tag}] {test.name}: OFF={off} ON={on}", flush=True)
    return matched, no_baseline, divergences


def main():
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Differential ON-vs-OFF gate for --lower-exceptions (#5075).")
    parser.add_argument("-e",
                        "--esbmc",
                        default=os.environ.get("ESBMC", "build/src/esbmc/esbmc"),
                        help="path to the esbmc binary (default: $ESBMC or build/src/esbmc/esbmc)")
    parser.add_argument("--root",
                        action="append",
                        dest="roots",
                        metavar="DIR",
                        help="regression subtree to scan (repeatable; default: esbmc-cpp + python)")
    parser.add_argument("--mode",
                        action="append",
                        dest="modes",
                        metavar="MODE",
                        help="test mode to include (repeatable; default: CORE). Use --all-modes "
                        "to include every mode.")
    parser.add_argument("--all-modes",
                        action="store_true",
                        help="include every test mode (CORE, KNOWNBUG, FUTURE, THOROUGH).")
    parser.add_argument("-j",
                        "--jobs",
                        type=int,
                        default=os.cpu_count() or 4,
                        help="parallel ESBMC invocations (default: number of CPUs)")
    parser.add_argument("-t",
                        "--timeout",
                        type=int,
                        default=120,
                        help="per-invocation timeout in seconds (default: 120)")
    parser.add_argument("--list",
                        action="store_true",
                        help="list the selected tests and exit without running ESBMC")
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="print each test's OFF/ON verdicts as it completes")
    args = parser.parse_args()

    roots = args.roots or ["regression/esbmc-cpp", "regression/python"]
    for root in roots:
        if not os.path.isdir(root):
            sys.exit(f"error: regression root not found: {root} "
                     f"(run from the ESBMC source root)")
    modes = None if args.all_modes else (args.modes or ["CORE"])

    tests = collect_tests(roots, modes)
    print(
        f"Selected {len(tests)} exception-bearing test(s) "
        f"({'all modes' if modes is None else '/'.join(modes)}) "
        f"under {', '.join(roots)}.",
        flush=True)

    if args.list:
        for test in tests:
            print(f"  {test.name}  [{test.mode}]  {test.source}  {test.args}")
        return 0

    if not os.path.exists(args.esbmc):
        sys.exit(f"error: esbmc binary not found: {args.esbmc}")

    matched, no_baseline, divergences = run_suite(tests, args)

    print()
    print(f"matched (ON==OFF, both verdicts): {matched}")
    print(f"no-baseline (both error/timeout): {len(no_baseline)}")
    print(f"divergences:                      {len(divergences)}")

    if no_baseline:
        # Surface these so a reviewer can confirm they are genuinely
        # incomparable (e.g. a timeout) rather than a masked regression.
        print("\nno-baseline tests (both paths produced no verdict):")
        for test, off, on in sorted(no_baseline, key=lambda d: d[0].name):
            print(f"  {test.name}: OFF={off} ON={on}")

    if divergences:
        print("\nDIVERGENCES (lowered path disagrees with imperative path):")
        for test, off, on in sorted(divergences, key=lambda d: d[0].name):
            print(f"  {test.name}: OFF={off} ON={on}")
        print("\nGATE FAILED — --lower-exceptions is not yet at parity.")
        return 1

    print("\nGATE PASSED — lowered path matches imperative path on all tests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
