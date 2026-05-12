#!/usr/bin/env python3
"""
Find the smallest --unwind N that lets a regression test still match its
expected output regexes, after removing --incremental-bmc and related
incremental-strategy flags from line 3 of test.desc.

Usage:
    find_min_unwind.py [--esbmc PATH] [--max-unwind N] [--apply] [--quiet]
                       TEST_DIR [TEST_DIR ...]

Per test:
  1. Reads test.desc (THOROUGH/CORE/FUTURE/KNOWNBUG, source, args, regex...).
  2. Skips if --incremental-bmc absent or test is in a suite that is
     legitimately about the incremental strategy.
  3. Strips --incremental-bmc, --unlimited-k-steps, --k-step N, --k-induction
     from the arg line.
  4. Tries --unwind in [1, 2, 4, 8, 16, 32, 64], runs esbmc, checks every
     expected regex matches stdout.
  5. Picks the smallest N. With --apply, rewrites test.desc; otherwise just
     reports.

Output columns: STATUS  N  TEST_DIR.  STATUS is one of:
    OK     - replacement found at unwind N
    SKIP   - test does not need rewriting
    KEEP   - no N <= max produced a matching verdict; leave test alone
    ERROR  - parser/runtime error
"""
import argparse
import multiprocessing as mp
import os
import re
import shlex
import subprocess
import sys

INCREMENTAL_FLAGS_PATTERN = re.compile(
    r"\s*--(?:incremental-bmc|unlimited-k-steps|k-induction|falsification)\b"
)
K_STEP_PATTERN = re.compile(r"\s*--(?:max-k-step|k-step)\s+\S+")
UNWIND_PATTERN = re.compile(r"\s*--unwind\s+\S+")

# Suites that are legitimately about an incremental strategy and must keep
# their flags. Match by path component.
LEGIT_SUITES = {
    "incremental-smt",
    "k-induction",
    "k-induction-parallel",
    "parallel-solving",
    "falsification",
}

DEFAULT_UNWINDS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]


def parse_desc(path):
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    if len(lines) < 4:
        raise ValueError(f"{path}: fewer than 4 lines")
    src = lines[1].strip()
    args = lines[2].strip()
    regex = [ln.strip() for ln in lines[3:] if ln.strip()]
    return src, args, regex


def is_legit_suite(test_dir):
    parts = os.path.normpath(test_dir).split(os.sep)
    return any(p in LEGIT_SUITES for p in parts)


def strip_incremental(args):
    out = INCREMENTAL_FLAGS_PATTERN.sub("", args)
    out = K_STEP_PATTERN.sub("", out)
    return " ".join(out.split())


def run_esbmc(esbmc, test_dir, src, args, unwind, timeout):
    arg_list = [esbmc] + [t for t in shlex.split(args) if t]
    if unwind is not None:
        arg_list += ["--unwind", str(unwind)]
    arg_list.append(src)
    try:
        p = subprocess.run(
            arg_list,
            cwd=test_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return None


def try_test(esbmc, test_dir, max_unwind, timeout):
    src, args, regex = parse_desc(os.path.join(test_dir, "test.desc"))
    if "--incremental-bmc" not in args:
        return ("SKIP", None, "no --incremental-bmc")
    if is_legit_suite(test_dir):
        return ("SKIP", None, "legit incremental suite")
    new_args = strip_incremental(args)
    # For FAILED-only tests, --no-unwinding-assertions ensures the FAILED
    # verdict comes from the property under test, not from an unwinding
    # assertion violation introduced by a small --unwind N.
    expected = "\n".join(regex)
    failure_only = "FAILED" in expected and "SUCCESSFUL" not in expected
    if failure_only and "--no-unwinding-assertions" not in new_args:
        new_args = (new_args + " --no-unwinding-assertions").strip()
    if UNWIND_PATTERN.search(new_args):
        candidates = [None]
    else:
        candidates = [n for n in DEFAULT_UNWINDS if n <= max_unwind]
    for n in candidates:
        out = run_esbmc(esbmc, test_dir, src, new_args, n, timeout)
        if out is None:
            continue
        if all(re.search(rx, out, re.MULTILINE) for rx in regex):
            return ("OK", n, new_args)
    return ("KEEP", None, "no unwind in range matched")


def apply_rewrite(test_dir, n, new_args):
    desc = os.path.join(test_dir, "test.desc")
    with open(desc) as f:
        lines = f.readlines()
    if n is None:
        lines[2] = new_args.strip() + "\n"
    else:
        lines[2] = f"{new_args} --unwind {n}".strip() + "\n"
    with open(desc, "w") as f:
        f.writelines(lines)


def _worker(job):
    esbmc, test_dir, max_unwind, timeout = job
    try:
        return (test_dir,) + try_test(esbmc, test_dir, max_unwind, timeout)
    except Exception as e:
        return (test_dir, "ERROR", None, str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esbmc", default=os.environ.get("ESBMC", "esbmc"))
    ap.add_argument("--max-unwind", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("test_dirs", nargs="+")
    args = ap.parse_args()

    jobs = [(args.esbmc, d, args.max_unwind, args.timeout) for d in args.test_dirs]
    rc = 0
    with mp.Pool(processes=args.jobs) as pool:
        for test_dir, status, n, detail in pool.imap_unordered(_worker, jobs, chunksize=1):
            if status == "ERROR":
                rc = 1
            if status == "OK" and args.apply:
                apply_rewrite(test_dir, n, detail)
            if not args.quiet or status not in ("OK", "SKIP"):
                print(f"{status:<5} {str(n) if n is not None else '-':<3} {test_dir}  # {detail}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
