#!/usr/bin/env python3
"""Differential sweep of a regression suite under an extra ESBMC flag.

An invariant-inference mode replaces the program with an abstraction, so its
failure modes are invisible to a test that only runs the mode it was written
for: a wrong verdict shows up as a *changed* verdict on a corpus, not as a
failing assertion. Both unsound verdicts found in --houdini-loop-invariants
were found this way and by nothing else, so this is a gate rather than a
one-off.

Verdicts are read from stdout, so a run that prints two of them (a strategy
emitting its own verdict after do_bmc has already emitted one) is reported as
whichever comes first in the precedence below -- deliberately, because
parse_result() in scripts/competitions/svcomp/esbmc-wrapper.py reads the
output the same way.

  scripts/loop-invariant-sweep.py --suite regression/k-induction \\
      --flag=--houdini-loop-invariants --out sweep.json --strip-drivers

Note the `--flag=` spelling: a value that itself starts with `--` is otherwise
read by argparse as another option.

Add --strip-drivers to remove the suite's own strategy flags first, which is
what you want when the flag under test owns the run itself.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import shlex
import subprocess
import sys

DRIVER_FLAGS = {
    "--k-induction",
    "--k-induction-parallel",
    "--incremental-bmc",
    "--falsification",
    "--termination",
    "--loop-invariant",
    "--loop-invariant-check",
    "--synthesise-loop-invariants",
    "--houdini-loop-invariants",
    "--incremental-context-bound",
}

VERDICTS = ("SUCCESSFUL", "FAILED", "UNKNOWN")


def verdict(out):
    """First verdict line in the output, matching how the SV-COMP wrapper reads it."""
    best, at = "NONE", len(out) + 1
    for v in VERDICTS:
        m = re.search(rf"^VERIFICATION {v}$", out, re.M)
        if m and m.start() < at:
            best, at = v, m.start()
    return best


def expected(desc_lines):
    body = "\n".join(desc_lines[3:])
    for v in VERDICTS:
        if "VERIFICATION " + v in body:
            return v
    return "?"


def run(esbmc, cwd, args, timeout):
    try:
        # check=False: a non-zero exit is the normal outcome for a failing
        # verdict, and the verdict is read from the output either way.
        p = subprocess.run([esbmc] + args, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, check=False)
        return verdict(p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def one(args, name):
    d = os.path.join(args.suite, name)
    desc = os.path.join(d, "test.desc")
    if not os.path.isfile(desc):
        return None
    with open(desc, encoding="utf-8") as f:
        lines = f.read().splitlines()
    if len(lines) < 3:
        return None
    src, flags = lines[1].strip(), shlex.split(lines[2])
    kept = [a for a in flags if a not in DRIVER_FLAGS] \
        if args.strip_drivers else flags
    return {
        "test": name,
        "kind": lines[0].strip(),
        "expected": expected(lines),
        "baseline": run(args.esbmc, d, [src] + flags, args.timeout),
        "flagged": run(args.esbmc, d, [src] + kept + [args.flag], args.timeout),
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--flag", required=True)
    ap.add_argument("--esbmc", default="build/src/esbmc/esbmc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--strip-drivers", action="store_true")
    args = ap.parse_args()

    # Each run uses the test's own directory as cwd, so a relative binary or
    # suite path would resolve against that instead of the invocation.
    args.esbmc = os.path.abspath(args.esbmc)
    args.suite = os.path.abspath(args.suite)
    args.out = os.path.abspath(args.out)
    return args


def collect(args):
    names = sorted(n for n in os.listdir(args.suite) if os.path.isdir(os.path.join(args.suite, n)))
    rows = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for r in ex.map(lambda n: one(args, n), names):
            if r:
                rows.append(r)
    return rows


def summarise(rows):
    # A program the suite expects to fail that the flag reports proved is the
    # only outcome that is categorically wrong; everything else is precision.
    unsound = [r for r in rows if r["expected"] == "FAILED" and r["flagged"] == "SUCCESSFUL"]
    alarms = [r for r in rows if r["expected"] == "SUCCESSFUL" and r["flagged"] == "FAILED"]
    gained = [
        r for r in rows if r["baseline"] != "SUCCESSFUL" and r["flagged"] == "SUCCESSFUL"
        and r["expected"] == "SUCCESSFUL"
    ]
    return unsound, alarms, gained


def main():
    args = parse_args()
    rows = collect(args)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)

    unsound, alarms, gained = summarise(rows)
    print(f"{len(rows)} tests  unsound={len(unsound)} "
          f"false-alarms={len(alarms)} gained={len(gained)}")
    for r in unsound:
        print("  UNSOUND:", r["test"])
    return 1 if unsound else 0


if __name__ == "__main__":
    sys.exit(main())
