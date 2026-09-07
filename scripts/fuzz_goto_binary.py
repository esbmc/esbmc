#!/usr/bin/env python3
"""Truncate and corrupt a goto-binary, and report any run that hangs or dies
from a signal.

Both readers -- CBMC's (`0x7f GBF`) and ESBMC's own (`GBF`) -- parse untrusted
input, so a malformed file must produce a diagnostic, never a crash and never an
unbounded loop. Five defects were found this way that inspection had missed:
an EOF that left no error flag, an `int` stored in a `char` so EOF never matched
the terminator (an infinite loop), two `abort()`s on conditions that describe
the input rather than an ESBMC bug, and a 32-bit overflow in a table resize.

  scripts/fuzz_goto_binary.py build/src/esbmc/esbmc prog.goto
  scripts/fuzz_goto_binary.py build/src/esbmc/esbmc main.goto --extra library.goto

Exits non-zero if anything hangs or is killed by a signal.
"""

import argparse
import pathlib
import random
import subprocess
import sys
import tempfile

# Exit codes a well-behaved run may use: verified, falsified, usage/parse error,
# and EX_SOFTWARE for an internal error that was still reported cleanly.
CLEAN_EXITS = {0, 1, 6, 70}


def run(esbmc, path, extra, timeout):
    """Return "ok", "hang", or "signal N". Output is captured as bytes: a
    corrupted binary makes ESBMC echo raw bytes, and decoding them as text
    aborts the sweep rather than the run under test."""
    cmd = [esbmc, str(path)] + (["--binary"] + extra if extra else ["--binary"])
    try:
        # check=False: a non-zero exit is what this sweep reads, not an error.
        r = subprocess.run(cmd, capture_output=True, timeout=timeout,
                           check=False)
    except subprocess.TimeoutExpired:
        return "hang"
    if r.returncode < 0:
        return f"signal {-r.returncode}"
    if r.returncode not in CLEAN_EXITS:
        return f"exit {r.returncode}"
    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("esbmc")
    ap.add_argument("binary")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="further --binary arguments, e.g. a library goto")
    ap.add_argument("--step", type=int, default=53,
                    help="truncate every STEP bytes")
    ap.add_argument("--corruptions", type=int, default=40)
    ap.add_argument("--flips", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--timeout", type=int, default=25)
    args = ap.parse_args()

    data = pathlib.Path(args.binary).read_bytes()
    random.seed(args.seed)
    bad = []

    with tempfile.TemporaryDirectory() as td:
        probe = pathlib.Path(td) / "probe.goto"

        for n in range(4, len(data), args.step):
            probe.write_bytes(data[:n])
            verdict = run(args.esbmc, probe, args.extra, args.timeout)
            if verdict != "ok":
                bad.append(f"truncated to {n} bytes: {verdict}")

        for i in range(args.corruptions):
            mutant = bytearray(data)
            for _ in range(args.flips):
                mutant[random.randrange(4, len(mutant))] = random.randrange(256)
            probe.write_bytes(bytes(mutant))
            verdict = run(args.esbmc, probe, args.extra, args.timeout)
            if verdict != "ok":
                bad.append(f"corruption #{i}: {verdict}")

    checked = len(range(4, len(data), args.step)) + args.corruptions
    print(f"{checked} malformed inputs checked, {len(bad)} bad")
    for b in bad:
        print("  " + b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
