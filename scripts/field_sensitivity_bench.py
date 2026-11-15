#!/usr/bin/env python3
"""Measure the #7521 field-sensitivity corpus (regression/field-sensitivity).

For each row, run ESBMC with Z3 and record symex time, SSA assignments, solver
time and peak memory; then dump the formula with --smtlib --output, record its
size and solve it with the Bitwuzla Python bindings (pip install bitwuzla).
Prints a Markdown table. Pass --flags to compare a mode such as
--field-sensitive-ssa against the same binary without it.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

CORPUS = Path(__file__).resolve().parent.parent / "regression" / "field-sensitivity"

PATTERNS = {
    "symex": r"Symex completed in: ([\d.]+)s \((\d+) assignments\)",
    "sliced": r"remaining after simplification \((\d+) assignments\)",
    "encode": r"Encoding to solver time: ([\d.]+)s",
    "solve": r"Runtime decision procedure: ([\d.]+)s",
    "verdict": r"^(VERIFICATION \w+)$",
}

BITWUZLA_SOLVE = """
import sys, time
import bitwuzla
p = bitwuzla.Parser(bitwuzla.TermManager(), bitwuzla.Options())
t = time.perf_counter()
p.parse(sys.argv[1], False, True)
print("time=%.3f" % (time.perf_counter() - t))
"""


def run(cmd, timeout):
    """Run cmd, return (stdout+stderr, wall seconds, peak RSS in MiB)."""
    start = time.perf_counter()
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        timer = threading.Timer(timeout, proc.kill)
        timer.start()
        out = proc.stdout.read()
        # wait4 rather than wait: it reports this child's own peak RSS.
        _, status, usage = os.wait4(proc.pid, 0)
        timer.cancel()
        proc.returncode = os.waitstatus_to_exitcode(status)
    return out, time.perf_counter() - start, usage.ru_maxrss / 1024
    """ESBMC command line for a corpus row, without its solver and timeout flags."""


def esbmc_cmd(binary, row, extra):
    desc = (row / "test.desc").read_text().splitlines()
    flags = [f for f in shlex.split(desc[2]) if f != "--z3"]
    if "--timeout" in flags:
        i = flags.index("--timeout")
        del flags[i:i + 2]
    return [binary, str(row / desc[1])] + flags + extra


def measure(binary, row, extra, bw_python, timeout):
    """Measure one corpus row; returns a dict of the recorded numbers."""
    res = {"row": row.name}
    out, res["wall"], res["rss"] = run(esbmc_cmd(binary, row, extra + ["--z3"]), timeout)
    for key, pat in PATTERNS.items():
        m = re.search(pat, out, re.M)
        res[key] = m.groups() if m else None
    with tempfile.TemporaryDirectory() as tmp:
        smt = os.path.join(tmp, "f.smt2")
        dump, _, _ = run(esbmc_cmd(binary, row, extra + ["--smtlib", "--output", smt]), timeout)
        if not os.path.exists(smt):
            res["bytes"] = res["bw"] = None
            res["bw_result"] = "no VCC" if "0 remaining" in dump else "no dump"
            return res
        res["bytes"] = os.path.getsize(smt)
        bw_out, _, _ = run([bw_python, "-c", BITWUZLA_SOLVE, smt], timeout)
        m = re.search(r"time=([\d.]+)", bw_out)
        res["bw"] = float(m.group(1)) if m else None
        res["bw_result"] = "sat" if "\nsat" in "\n" + bw_out else (
            "unsat" if "unsat" in bw_out else bw_out.strip()[-40:])
    return res


def fmt(res):
    """One Markdown table row."""

    def get(key, idx=0):
        v = res.get(key)
        return v[idx] if v else "-"

    return (f"| {res['row']} | {get('verdict')} | {get('symex')} | {get('symex', 1)} | "
            f"{get('sliced')} | {res['bytes']} | {get('solve')} | {res['bw']} "
            f"({res['bw_result']}) | {res['wall']:.2f} | {res['rss']:.0f} |")


def main():
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--esbmc", default="esbmc")
    ap.add_argument("--flags", default="", help="extra ESBMC flags for every row")
    ap.add_argument("--rows", default=".", help="regex selecting corpus rows")
    ap.add_argument("--bitwuzla-python",
                    default=sys.executable,
                    help="Python interpreter that can import bitwuzla")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    rows = sorted(p for p in CORPUS.iterdir() if p.is_dir() and re.search(args.rows, p.name))
    print("| row | verdict | symex s | SSA steps | after slicing | SMT-LIB bytes | "
          "Z3 s | Bitwuzla s | wall s | peak MiB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        res = measure(args.esbmc, row, shlex.split(args.flags), args.bitwuzla_python, args.timeout)
        print(fmt(res), flush=True)


if __name__ == "__main__":
    main()
