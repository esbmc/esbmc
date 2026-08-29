#!/usr/bin/env python3
"""Report how many SV-COMP tasks finished close enough to the limit to flip.

A run's score moves for two reasons: ESBMC got better or worse at something, or
tasks sitting a hair under the time limit crossed it. The second is not a
regression, but it looks exactly like one in the summary -- issue #6831 spent a
week distinguishing them for 131 Juliet tasks whose median runtime was 99.1 s of
a 100 s limit.

This reads BenchExec's per-run XML and splits the outcome into what a few
percent of runtime can and cannot change:

  * marginal wins    -- correct, but within <margin> of the limit; a slowdown
                        of that size loses them
  * marginal losses  -- timed out, but a speedup of that size would recover
                        them (only visible when the run records how long the
                        task actually took)
  * safe             -- everything else, which no plausible timing change moves

Usage:
    marginal_timeouts.py results.xml.bz2 [more.xml ...] [--margin 5]
"""

import argparse
import bz2
import sys
from xml.etree import ElementTree


def parse_seconds(value):
    """BenchExec writes times as '12.345678901s'."""
    if not value:
        return None
    try:
        return float(value.rstrip("s"))
    except ValueError:
        return None


def load(path):
    """Read a BenchExec result file, transparently un-bzipping it."""
    opener = bz2.open if path.endswith(".bz2") else open
    with opener(path, "rb") as handle:
        return ElementTree.parse(handle).getroot()


def limit_of(result, override):
    """The run's time budget in seconds: the override, else what BenchExec recorded."""
    if override:
        return override
    # BenchExec records the budget on the result element; cputime is what the
    # limit applies to, so prefer it and fall back to the wall clock one.
    for attribute in ("cpuTimelimit", "timelimit", "walltimelimit"):
        seconds = parse_seconds(result.get(attribute))
        if seconds:
            return seconds
    return None


def classify(result, limit, margin):
    """Split one result file's runs into marginal wins, marginal losses, safe."""
    threshold = limit * (1.0 - margin / 100.0)
    wins, losses, safe = [], [], 0

    for run in result.iter("run"):
        columns = {c.get("title"): c.get("value") for c in run.findall("column")}
        status = columns.get("status", "")
        cputime = parse_seconds(columns.get("cputime"))
        name = run.get("name") or run.get("files") or "?"

        if status.startswith("TIMEOUT"):
            losses.append((name, cputime))
        elif cputime is not None and cputime >= threshold:
            wins.append((name, cputime))
        else:
            safe += 1

    return wins, losses, safe


def main():
    """Print the marginal band for every result file named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="BenchExec results XML (.xml or .xml.bz2)")
    parser.add_argument("--margin", type=float, default=5.0,
                        help="percent of the limit that counts as marginal (default: 5)")
    parser.add_argument("--limit", type=float, help="time limit in seconds, if not in the XML")
    parser.add_argument("--list", action="store_true", help="name every marginal task")
    opts = parser.parse_args()

    total_wins, total_losses, total_safe = [], [], 0
    for path in opts.results:
        result = load(path)
        limit = limit_of(result, opts.limit)
        if limit is None:
            print(f"{path}: no time limit recorded; pass --limit", file=sys.stderr)
            return 2
        wins, losses, safe = classify(result, limit, opts.margin)
        total_wins += wins
        total_losses += losses
        total_safe += safe
        print(f"{path}: limit {limit:g}s, marginal band {limit * (1 - opts.margin / 100):g}-{limit:g}s")

    print(f"\nmarginal wins   {len(total_wins):>6}  correct, but a {opts.margin:g}% slowdown loses them")
    print(f"marginal losses {len(total_losses):>6}  timed out")
    print(f"safe            {total_safe:>6}")

    if opts.list:
        print()
        for name, cputime in sorted(total_wins, key=lambda w: -(w[1] or 0)):
            print(f"  {cputime:>8.1f}s  {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
