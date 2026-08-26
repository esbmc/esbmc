#!/usr/bin/env python3
"""Measure what --proof-cache costs and saves on SV-COMP tasks.

This is a measurement tool, NOT part of the competition path: it drives
esbmc-wrapper.py in --dry-run mode to obtain the exact command a competition
run would use, then times that command three ways per task.

  baseline  no cache at all -- the competition configuration, plus
            --multi-property, which the cache needs and which therefore has to
            be on all three legs for the comparison to mean anything
  cold      a fresh cache; nothing can hit, so this is the overhead a
            single-shot run would pay (the number that decides whether the
            wrapper should ever enable it)
  warm      the same cache a second time -- the upper bound on what
            re-verifying an unchanged task can save

Every task must reach the same verdict all three ways. A disagreement is a
soundness alarm, reported as MISMATCH and counted separately; it is not a
performance result.

Each task gets its own cache directory by default, which compares a task only
against itself. --shared-cache puts every task in one directory instead, the
way a persisted CI cache works, so a key two *different* tasks collide on shows
up as a MISMATCH. Only that mode can find a collision at all; a per-task cache
cannot (esbmc/esbmc#7143).

  ./proof-cache-bench.py -p reach.prp -a 64 -s kinduction task1.c task2.c ...
  ./proof-cache-bench.py -p reach.prp --shared-cache task1.c task2.c ...
"""

import argparse
import contextlib
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WRAPPER = os.path.join(HERE, "esbmc-wrapper.py")


def competition_command(task, prop, arch, strategy):
    """The command line a real competition run would execute for this task."""
    out = subprocess.run(
        [sys.executable, WRAPPER, "-n", "-p", prop, "-a", str(arch),
         "-s", strategy, task],
        capture_output=True, text=True, check=False).stdout
    for line in out.splitlines():
        if line.startswith("Command: "):
            return shlex.split(line[len("Command: "):])
    return None


def resolve_binary(cmd, esbmc):
    """The wrapper emits `./esbmc`, which only resolves inside a competition
    archive. Point it at the binary actually under test."""
    if cmd and esbmc:
        return [esbmc] + cmd[1:]
    return cmd


def timed(cmd, timeout):
    """Run one command, returning its wall time, verdict and reuse counts."""
    start = time.monotonic()
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout,
                           text=True, check=False)
    except subprocess.TimeoutExpired:
        return None
    out = r.stdout + r.stderr
    verdict = re.search(r"VERIFICATION (\w+)", out)
    reuse = re.search(r"Proof cache: (\d+) claim\(s\) reused, (\d+) solved", out)
    return {"wall": time.monotonic() - start,
            "verdict": verdict.group(1) if verdict else "NONE",
            "reused": int(reuse.group(1)) if reuse else 0,
            "solved": int(reuse.group(2)) if reuse else 0}


def median_run(cmd, timeout, repeats):
    """Run a command `repeats` times, taking the median wall time."""
    runs = [timed(cmd, timeout) for _ in range(repeats)]
    if any(r is None for r in runs):
        return None
    return {"wall": statistics.median(r["wall"] for r in runs),
            "verdict": runs[0]["verdict"],
            "reused": runs[0]["reused"],
            "solved": runs[0]["solved"]}


def measure(task, args, shared=None):
    """Time one task three ways. Returns a row, or None when it cannot run.

    `shared` is a cache directory carried across tasks; without it the task
    gets a private one that is discarded afterwards.
    """
    cmd = resolve_binary(
        competition_command(task, args.propertyfile, args.arch, args.strategy),
        args.esbmc)
    if cmd is None:
        return None

    # The cache is wired into the per-claim solve alone, and ESBMC refuses
    # --proof-cache without --multi-property. The competition command does not
    # use it, so it goes on all three legs: a baseline that measures a
    # different configuration from the cached runs measures nothing.
    if "--multi-property" not in cmd:
        cmd = cmd + ["--multi-property"]

    with contextlib.ExitStack() as stack:
        cache = shared or stack.enter_context(tempfile.TemporaryDirectory())
        runs = {
            "base": median_run(cmd, args.timeout, args.repeats),
            "cold": median_run(cmd + ["--proof-cache", cache], args.timeout, 1),
            "warm": median_run(cmd + ["--proof-cache", cache], args.timeout,
                               args.repeats),
        }

    if any(r is None for r in runs.values()):
        return None
    runs["name"] = os.path.basename(task)[:38]
    return runs


def report(rows):
    """Print the per-task table and the totals. Returns the mismatch count."""
    print(f"{'task':<40}{'baseline':>10}{'cold':>10}{'warm':>10}"
          f"{'reused':>10}  verdict")
    totals = {"base": 0.0, "cold": 0.0, "warm": 0.0}
    counted = 0
    mismatches = []

    for row in rows:
        base, cold, warm = row["base"], row["cold"], row["warm"]
        verdicts = (base["verdict"], cold["verdict"], warm["verdict"])
        if len(set(verdicts)) != 1:
            mismatches.append((row["name"],) + verdicts)
            print(f"{row['name']:<40}{'':>30}  MISMATCH {'/'.join(verdicts)}")
            continue

        counted += 1
        for key in totals:
            totals[key] += row[key]["wall"]
        reused = f"{warm['reused']}/{warm['reused'] + warm['solved']}"
        print(f"{row['name']:<40}{base['wall']:>10.2f}{cold['wall']:>10.2f}"
              f"{warm['wall']:>10.2f}{reused:>10}  {base['verdict']}")

    if counted and not any(
            row["warm"]["reused"] or row["warm"]["solved"] for row in rows):
        print("\nNOTE: no run reported a cache line, so --proof-cache never "
              "activated -- ESBMC says why on the run's first line. These "
              "timings say nothing about the cache.")

    print(f"\ntasks measured: {counted}, MISMATCHES: {len(mismatches)}")
    for name, *verdicts in mismatches:
        print(f"  MISMATCH {name}: baseline={verdicts[0]} "
              f"cold={verdicts[1]} warm={verdicts[2]}")

    if counted:
        base, cold, warm = totals["base"], totals["cold"], totals["warm"]
        print(f"  baseline {base:8.2f}s")
        print(f"  cold     {cold:8.2f}s   overhead {100 * (cold / base - 1):+.1f}%"
              f"  <- what a single-shot competition run would pay")
        print(f"  warm     {warm:8.2f}s   saved    {100 * (1 - warm / base):+.1f}%"
              f"  <- re-verifying an unchanged task")
    return len(mismatches)


def main():
    """Measure every task named on the command line."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--propertyfile", required=True)
    ap.add_argument("-a", "--arch", type=int, choices=[32, 64], default=32)
    ap.add_argument("-s", "--strategy",
                    choices=["kinduction", "falsi", "incr", "fixed"],
                    default="kinduction")
    ap.add_argument("-t", "--timeout", type=int, default=300)
    ap.add_argument("-r", "--repeats", type=int, default=3)
    ap.add_argument("--esbmc", default=shutil.which("esbmc"),
                    help="binary to test (default: esbmc on PATH); the "
                         "wrapper itself always names ./esbmc")
    ap.add_argument("--shared-cache", action="store_true",
                    help="one cache directory for every task, so a key two "
                         "different tasks collide on shows up as a MISMATCH")
    ap.add_argument("tasks", nargs="+")
    args = ap.parse_args()

    with contextlib.ExitStack() as stack:
        shared = (stack.enter_context(tempfile.TemporaryDirectory())
                  if args.shared_cache else None)
        rows = [row for row in (measure(t, args, shared) for t in args.tasks)
                if row]
    skipped = len(args.tasks) - len(rows)
    if skipped:
        print(f"(skipped {skipped} task(s) that timed out or would not run)")
    return 1 if report(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
