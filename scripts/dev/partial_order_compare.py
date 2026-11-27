#!/usr/bin/env python3
"""
Compare --partial-order with the default exploration on regression tests
(#10990, phase 4).

Each test runs twice, with its test.desc arguments as written and with
--partial-order appended. Only tests whose sources include <pthread.h> run,
and the partial_order_* tests are skipped, since they pin --partial-order
already.

Usage:
    partial_order_compare.py [--esbmc PATH] [--timeout S] [--jobs N]
                             [--csv FILE] SUITE_DIR [SUITE_DIR ...]

Each run is classified as SUCCESSFUL, FAILED, UNKNOWN, REJECTED (the scope
check or a flag conflict refused the input), TIMEOUT or ERROR. The summary
counts the tests both modes decide, how many of those agree, and lists every
disagreement with the verdict test.desc expects. A disagreement fails the
script, so it can gate a change.
"""
import argparse
import collections
import csv
import multiprocessing as mp
import os
import re
import shlex
import subprocess
import sys
import time

VERDICTS = (
    ("VERIFICATION SUCCESSFUL", "SUCCESSFUL"),
    ("VERIFICATION FAILED", "FAILED"),
    ("VERIFICATION UNKNOWN", "UNKNOWN"),
)
REJECTION = re.compile(r"--partial-order (?:does not support (.*?) yet"
                       r"|cannot be combined with (--\S+))")


def parse_desc(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    if len(lines) < 3:
        return None
    expected = None
    for line in lines[3:]:
        for text, verdict in VERDICTS:
            if text in line:
                expected = verdict
    return lines[0].strip(), lines[1].strip(), lines[2].strip(), expected


def uses_pthreads(test_dir):
    for name in os.listdir(test_dir):
        if name.endswith((".c", ".cpp", ".h")):
            path = os.path.join(test_dir, name)
            with open(path, encoding="utf-8", errors="replace") as f:
                if "pthread.h" in f.read():
                    return True
    return False


def classify(returncode, output):
    for text, verdict in VERDICTS:
        if text in output:
            return verdict, ""
    match = REJECTION.search(output)
    if match:
        return "REJECTED", match.group(1) or match.group(2)
    return "ERROR", f"exit {returncode}"


def run(esbmc, test_dir, src, args, timeout):
    cmd = [esbmc, src] + shlex.split(args)
    start = time.monotonic()
    try:
        proc = subprocess.run(cmd,
                              cwd=test_dir,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              timeout=timeout,
                              check=False)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "", timeout
    elapsed = time.monotonic() - start
    output = proc.stdout.decode("utf-8", errors="replace")
    verdict, detail = classify(proc.returncode, output)
    return verdict, detail, elapsed


def compare(job):
    esbmc, test_dir, timeout = job
    level, src, args, expected = parse_desc(os.path.join(test_dir, "test.desc"))
    default = run(esbmc, test_dir, src, args, timeout)
    partial = run(esbmc, test_dir, src, args + " --partial-order", timeout)
    return test_dir, level, expected, default, partial


def collect(suites):
    tests = []
    for suite in suites:
        for name in sorted(os.listdir(suite)):
            test_dir = os.path.join(suite, name)
            desc = os.path.join(test_dir, "test.desc")
            if name.startswith("partial_order_") or not os.path.isfile(desc):
                continue
            if parse_desc(desc) and uses_pthreads(test_dir):
                tests.append(test_dir)
    return tests


DECIDED = ("SUCCESSFUL", "FAILED")


def print_status(label, statuses):
    counts = collections.Counter(statuses)
    print(f"{label}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


def print_median_times(both):
    times_d = sorted(r[3][2] for r in both)
    times_p = sorted(r[4][2] for r in both)
    print(f"median time on those: default {times_d[len(both) // 2]:.2f} s, "
          f"--partial-order {times_p[len(both) // 2]:.2f} s")


def print_rejections(results):
    reasons = collections.Counter(r[4][1] for r in results if r[4][0] == "REJECTED")
    print("rejections:")
    for reason, count in reasons.most_common():
        print(f"  {count:4d}  {reason}")


def summarise(results):
    both = [r for r in results if r[3][0] in DECIDED and r[4][0] in DECIDED]
    disagree = [r for r in both if r[3][0] != r[4][0]]

    print(f"tests: {len(results)}")
    print_status("--partial-order", (r[4][0] for r in results))
    print_status("default", (r[3][0] for r in results))
    print(f"decided by both: {len(both)}, agreeing: {len(both) - len(disagree)}")
    if both:
        print_median_times(both)
    print_rejections(results)
    for test_dir, level, expected, default, partial in disagree:
        print(f"DISAGREE {test_dir} ({level}, expects {expected}): "
              f"default {default[0]}, --partial-order {partial[0]}")
    return not disagree


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--esbmc", default="esbmc")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--csv")
    parser.add_argument("suites", nargs="+")
    opts = parser.parse_args()

    esbmc = os.path.abspath(opts.esbmc) if os.sep in opts.esbmc else opts.esbmc
    jobs = [(esbmc, t, opts.timeout) for t in collect(opts.suites)]
    with mp.Pool(opts.jobs) as pool:
        results = sorted(pool.imap_unordered(compare, jobs))

    if opts.csv:
        with open(opts.csv, "w", newline="", encoding="utf-8") as f:
            out = csv.writer(f)
            out.writerow([
                "test", "level", "expected", "default", "default_s", "partial_order",
                "partial_order_s", "detail"
            ])
            for test_dir, level, expected, default, partial in results:
                out.writerow([
                    test_dir, level, expected, default[0], f"{default[2]:.2f}", partial[0],
                    f"{partial[2]:.2f}", partial[1]
                ])
    return 0 if summarise(results) else 1


if __name__ == "__main__":
    sys.exit(main())
