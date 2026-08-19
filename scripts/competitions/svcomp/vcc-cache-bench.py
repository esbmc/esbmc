#!/usr/bin/env python3
"""Measure what --vcc-cache costs and saves on SV-COMP tasks.

This is a measurement tool, NOT part of the competition path: it drives
esbmc-wrapper.py in --dry-run mode to obtain the exact command a competition
run would use, then times that command three ways per task.

  baseline  no cache at all -- the competition configuration
  cold      a fresh cache; nothing can hit, so this is the overhead a
            single-shot run would pay (the number that decides whether the
            wrapper should ever enable it)
  warm      the same cache a second time -- the upper bound on what
            re-verifying an unchanged task can save

Every task must reach the same verdict all three ways. A disagreement is a
soundness alarm, reported as MISMATCH and counted separately; it is not a
performance result.

  ./vcc-cache-bench.py -p reach.prp -a 64 -s kinduction task1.c task2.c ...
"""

import argparse
import os
import re
import shlex
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
        capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Command: "):
            return shlex.split(line[len("Command: "):])
    return None


def timed(cmd, timeout):
    start = time.monotonic()
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
    except subprocess.TimeoutExpired:
        return None
    out = r.stdout + r.stderr
    verdict = re.search(r"VERIFICATION (\w+)", out)
    reuse = re.search(r"VCC cache: (\d+) claim\(s\) reused, (\d+) solved", out)
    return dict(wall=time.monotonic() - start,
                verdict=verdict.group(1) if verdict else "NONE",
                reused=int(reuse.group(1)) if reuse else 0,
                solved=int(reuse.group(2)) if reuse else 0)


def median_run(cmd, timeout, repeats):
    runs = [timed(cmd, timeout) for _ in range(repeats)]
    if any(r is None for r in runs):
        return None
    return dict(wall=statistics.median(r["wall"] for r in runs),
                verdict=runs[0]["verdict"],
                reused=runs[0]["reused"], solved=runs[0]["solved"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--propertyfile", required=True)
    ap.add_argument("-a", "--arch", type=int, choices=[32, 64], default=32)
    ap.add_argument("-s", "--strategy",
                    choices=["kinduction", "falsi", "incr", "fixed"],
                    default="kinduction")
    ap.add_argument("-t", "--timeout", type=int, default=300)
    ap.add_argument("-r", "--repeats", type=int, default=3)
    ap.add_argument("tasks", nargs="+")
    args = ap.parse_args()

    print(f"{'task':<40}{'baseline':>10}{'cold':>10}{'warm':>10}"
          f"{'reused':>10}  verdict")
    totals = dict(base=0.0, cold=0.0, warm=0.0)
    counted = skipped = 0
    mismatches = []

    for task in args.tasks:
        cmd = competition_command(task, args.propertyfile, args.arch,
                                  args.strategy)
        if cmd is None:
            skipped += 1
            continue

        with tempfile.TemporaryDirectory() as cache:
            base = median_run(cmd, args.timeout, args.repeats)
            cold = median_run(cmd + ["--vcc-cache", cache], args.timeout, 1)
            warm = median_run(cmd + ["--vcc-cache", cache], args.timeout,
                              args.repeats)

        if not base or not cold or not warm:
            skipped += 1
            continue

        name = os.path.basename(task)[:38]
        if not base["verdict"] == cold["verdict"] == warm["verdict"]:
            mismatches.append((name, base["verdict"], cold["verdict"],
                               warm["verdict"]))
            print(f"{name:<40}{'':>30}  MISMATCH "
                  f"{base['verdict']}/{cold['verdict']}/{warm['verdict']}")
            continue

        counted += 1
        totals["base"] += base["wall"]
        totals["cold"] += cold["wall"]
        totals["warm"] += warm["wall"]
        print(f"{name:<40}{base['wall']:>10.2f}{cold['wall']:>10.2f}"
              f"{warm['wall']:>10.2f}"
              f"{str(warm['reused'])+'/'+str(warm['reused']+warm['solved']):>10}"
              f"  {base['verdict']}")

    print(f"\ntasks measured: {counted}, skipped: {skipped}, "
          f"MISMATCHES: {len(mismatches)}")
    for m in mismatches:
        print(f"  MISMATCH {m[0]}: baseline={m[1]} cold={m[2]} warm={m[3]}")
    if counted:
        b, c, w = totals["base"], totals["cold"], totals["warm"]
        print(f"  baseline {b:8.2f}s")
        print(f"  cold     {c:8.2f}s   overhead {100 * (c / b - 1):+.1f}%  "
              f"<- what a single-shot competition run would pay")
        print(f"  warm     {w:8.2f}s   saved    {100 * (1 - w / b):+.1f}%  "
              f"<- re-verifying an unchanged task")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
