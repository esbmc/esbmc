#!/usr/bin/env python3
"""Interleaved A/B measurement of two ESBMC binaries on one input.

The measurement host drifts by more than the effects worth chasing (issue
#6831's multiplicative term is ~3.5 %; a quiet workstation moves ±15 % over
tens of minutes), so timing one binary and then the other attributes drift to
the change. The two binaries therefore alternate inside a single loop, the
order flips every pair, and the reported ratio is the median of the per-pair
ratios -- pairing cancels drift that a ratio of two independent medians keeps.

Counts are the other half. A commit that moves VCCs or symex assignments has
changed what the solver is asked to prove; one that moves only wall time has
not, which is the distinction the #6831 W0 bisect stops on.

  scripts/perf/ab_interleave.py --a old/esbmc --b new/esbmc --pairs 12 \\
      -- scripts/perf/oracles/loop10k.c --unwind 10000 --overflow-check --quiet

Exit status is 2 if any run failed, so a driver script cannot mistake a
crashed build for a fast one.
"""

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

# Ordered for printing. wall contains everything; bmc contains symex through
# solve; goto is outside bmc. They are not disjoint and must not be summed.
PHASES = (
    ("wall", None),  # measured here, not parsed
    ("goto", re.compile(r"GOTO program creation time: ([0-9.]+)s")),
    ("goto-proc", re.compile(r"GOTO program processing time: ([0-9.]+)s")),
    ("symex", re.compile(r"Symex completed in: ([0-9.]+)s")),
    ("caching", re.compile(r"Caching time: ([0-9.]+)s")),
    ("slicing", re.compile(r"Slicing time: ([0-9.]+)s")),
    ("encoding", re.compile(r"Encoding to solver time: ([0-9.]+)s")),
    ("solve", re.compile(r"Runtime decision procedure: ([0-9.]+)s")),
    ("bmc", re.compile(r"BMC program time: ([0-9.]+)s")),
)

# Only these two decide the verdict: they say what symex produced. "remaining"
# is post-simplification and moves when the simplifier changes, so it is
# reported but not judged on.
VERDICT_COUNTS = (
    ("assignments", re.compile(r"Symex completed in: [0-9.]+s \((\d+) assignments\)")),
    ("vccs", re.compile(r"Generated (\d+) VCC\(s\)")),
)
COUNTS = VERDICT_COUNTS + (("remaining", re.compile(r"Generated \d+ VCC\(s\), (\d+) remaining")), )


def parse(out):
    """Pull ESBMC's self-reported phases and counts out of one run's output.

    First match only: this assumes a single BMC run, which --k-induction and
    --incremental-bmc violate by reporting per iteration.
    """
    sample = {}
    for name, pattern in PHASES:
        if pattern is not None:
            found = pattern.search(out)
            sample[name] = float(found.group(1)) if found else None
    for name, pattern in COUNTS:
        found = pattern.search(out)
        sample[name] = int(found.group(1)) if found else None
    return sample


def run(binary, args, timeout, tmpdir):
    """Time one ESBMC run. Returns (sample, error); error is None on success."""
    env = dict(os.environ, TMPDIR=tmpdir)
    start = time.perf_counter()
    try:
        # ESBMC reports its phases on stderr, so the merge is load-bearing:
        # capturing stdout alone parses to all-None.
        proc = subprocess.run([binary] + args,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              timeout=timeout,
                              env=env,
                              check=False)
    except subprocess.TimeoutExpired:
        return None, f"{binary}: timed out after {timeout}s"
    except OSError as exc:
        return None, f"{binary}: {exc}"
    wall = time.perf_counter() - start

    if proc.returncode not in (0, 1):
        # 0 successful, 1 failed verification; anything else is not a result.
        tail = proc.stdout.decode(errors="replace").strip().splitlines()[-1:]
        return None, f"{binary}: exit {proc.returncode} ({' '.join(tail)})"

    sample = parse(proc.stdout.decode(errors="replace"))
    sample["wall"] = wall
    return sample, None


def ratios(a_samples, b_samples, name):
    """Per-pair B/A for one metric, over the pairs where both sides parsed."""
    out = []
    for a_sample, b_sample in zip(a_samples, b_samples):
        a_value, b_value = a_sample[name], b_sample[name]
        if a_value and b_value:
            out.append(b_value / a_value)
    return out


def print_timings(a_samples, b_samples):
    print(f"\n{'metric':<12}{'A':>10}{'B':>10}{'B/A':>9}{'IQR':>9}{'n':>4}")
    for name, _ in PHASES:
        pair_ratios = ratios(a_samples, b_samples, name)
        if not pair_ratios:
            continue
        a_median = statistics.median(s[name] for s in a_samples if s[name] is not None)
        b_median = statistics.median(s[name] for s in b_samples if s[name] is not None)
        spread = (statistics.quantiles(pair_ratios)[2] -
                  statistics.quantiles(pair_ratios)[0] if len(pair_ratios) > 3 else float("nan"))
        print(f"{name:<12}{a_median:>10.3f}{b_median:>10.3f}"
              f"{statistics.median(pair_ratios):>9.3f}{spread:>9.3f}{len(pair_ratios):>4}")
    print("wall > bmc > {symex, caching, slicing, encoding, solve}; goto is outside bmc")


def print_counts(a_samples, b_samples):
    """Print each count side by side; return the (moved, unknown) names."""
    moved, unknown = [], []
    for name, _ in COUNTS:
        a_values = {s[name] for s in a_samples if s[name] is not None}
        b_values = {s[name] for s in b_samples if s[name] is not None}
        if not a_values or not b_values:
            unknown.append(name)
            print(f"{name:<12}A={sorted(a_values)} B={sorted(b_values)}  NO DATA")
            continue
        differ = a_values != b_values
        if differ and name in dict(VERDICT_COUNTS):
            moved.append(name)
        print(f"{name:<12}A={sorted(a_values)} B={sorted(b_values)}"
              f"  {'DIFFER' if differ else 'identical'}")
    return moved, unknown


def summarise(a_samples, b_samples):
    print_timings(a_samples, b_samples)
    print()
    moved, unknown = print_counts(a_samples, b_samples)

    print()
    if unknown:
        print(f"no verdict: {', '.join(unknown)} did not parse on one or both sides")
    elif moved:
        print(f"counts moved ({', '.join(moved)}): the change alters what symex produces")
    else:
        print("counts identical: any time delta is not a change in symex's output")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="baseline binary")
    parser.add_argument("--b", required=True, help="binary under test")
    parser.add_argument("--pairs", type=int, default=12, help="A/B pairs, even (default: 12)")
    parser.add_argument("--timeout", type=float, default=900, help="per-run timeout in seconds")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="-- <input and ESBMC flags>")
    opts = parser.parse_args()

    esbmc_args = opts.args[1:] if opts.args[:1] == ["--"] else opts.args
    if not esbmc_args:
        parser.error("no input: pass the program and flags after --")
    # Odd pair counts leave one A/B block uncounterbalanced, which puts B a slot
    # later than A over the run and reintroduces the drift the flip removes.
    if opts.pairs < 2 or opts.pairs % 2:
        parser.error("--pairs must be an even number >= 2")

    # ESBMC extracts its bundled headers into TMPDIR on every run (~7 MB a
    # time), so give the children one of our own and take it away afterwards.
    tmpdir = tempfile.mkdtemp(prefix="ab-interleave-")
    a_samples, b_samples, errors = [], [], []
    try:
        # The first runs on a cold cache are ~15 % slower than the steady state
        # and decay non-linearly, which the order flip cannot cancel. Spend one
        # run per binary buying that transient off before measuring.
        for binary in (opts.a, opts.b):
            _, error = run(binary, esbmc_args, opts.timeout, tmpdir)
            if error:
                print(f"warm-up failed: {error}", file=sys.stderr)
                return 2

        for pair in range(opts.pairs):
            order = ["a", "b"] if pair % 2 == 0 else ["b", "a"]
            taken = {}
            for side in order:
                sample, error = run(getattr(opts, side), esbmc_args, opts.timeout, tmpdir)
                if error:
                    errors.append(f"pair {pair + 1}: {error}")
                taken[side] = sample
            # Both or neither: a half-pair would shift every later pair's
            # partner by one, which is exactly the pairing ratios() relies on.
            if taken["a"] and taken["b"]:
                a_samples.append(taken["a"])
                b_samples.append(taken["b"])
            print(f"pair {pair + 1}/{opts.pairs}", file=sys.stderr)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if a_samples:
        summarise(a_samples, b_samples)

    if errors:
        print("\nFAILED RUNS -- results above are incomplete:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
