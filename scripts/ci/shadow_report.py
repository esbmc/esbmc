#!/usr/bin/env python3
"""Measure what the fast lane would have missed (esbmc/esbmc#6735, rollout step 3).

Shadow mode runs the week's fast-lane subset alongside the full suite on every
PR and answers the two questions that decide whether the gate can be flipped:

* **How long does the fast lane actually take?** Projected wall-clock comes from
  durations measured elsewhere at a different parallelism; only running it says
  what it really costs.
* **What escapes it?** A test that failed in the full run but was never sampled
  is an escaped regression -- exactly the risk being traded away for speed.

A third answer falls out for free. A test the fast lane *ran and passed* that
then failed in the full run is neither caught nor escaped: it is flaky or
order-dependent. The issue calls flakiness out as the thing that will derail the
bisect loop, so it is counted separately rather than folded into either bucket.

``compare`` reports one PR; ``aggregate`` turns a pile of those reports into the
escaped-regression rate that rollout step 4 is supposed to be decided on.
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# pylint: disable=wrong-import-position
from ctest_timings import parse_junit  # noqa: E402
from select_tests import read_lines  # noqa: E402

SCHEMA = 1


def results(junit_path):
    """Read a JUnit report as ``{test: status}``."""
    return {name: status for name, _, status in parse_junit(junit_path)}


def compare_runs(fast, full, selection):
    """Classify every full-run failure against what the fast lane did.

    ``fast`` and ``full`` are ``{test: status}``; ``selection`` is what the week
    chose, which can name tests this build does not contain.
    """
    failed_full = {n for n, s in full.items() if s == "fail"}
    failed_fast = {n for n, s in fast.items() if s == "fail"}

    caught = sorted(failed_full & failed_fast)
    # Ran by the fast lane, passed there, failed in the full run.
    unstable = sorted(n for n in failed_full & set(fast) if n not in failed_fast)
    escaped = sorted(failed_full - set(fast))
    # Failed in the fast lane but not in the full run: unstable the other way.
    unstable += sorted(n for n in failed_fast - failed_full if n in full)

    return {
        "selected": len(selection),
        "fast_lane_ran": len(fast),
        "full_ran": len(full),
        "failures_full": len(failed_full),
        "caught": caught,
        "escaped": escaped,
        "unstable": sorted(set(unstable)),
    }


def escaped_rate(failures, escaped):
    """Fraction of real failures the fast lane never looked at."""
    return round(escaped / failures, 4) if failures else 0.0


def format_compare(report):
    """Render one PR's shadow result as Markdown."""
    failures = report["failures_full"]
    rate = escaped_rate(failures, len(report["escaped"]))
    wall = report["fast_lane_seconds"]
    lines = [
        "### Fast-lane shadow run",
        "",
        f"- fast lane: **{wall / 60:.1f} min** wall-clock, "
        f"{report['fast_lane_ran']} of {report['full_ran']} tests",
        f"- full run: {failures} failure(s)",
    ]
    if failures:
        lines.append(f"- caught by the fast lane: {len(report['caught'])}")
        lines.append(f"- **escaped**: {len(report['escaped'])} ({rate * 100:.0f}%)")
    else:
        lines.append("- nothing failed, so nothing could escape")
    if report["unstable"]:
        lines.append(f"- unstable (disagreed between the two runs): {len(report['unstable'])}")

    for label, key in (("Escaped", "escaped"), ("Unstable", "unstable")):
        if report[key]:
            lines += ["", f"<details><summary>{label}</summary>", ""]
            lines += [f"- `{n}`" for n in report[key][:50]]
            if len(report[key]) > 50:
                lines.append(f"- ... and {len(report[key]) - 50} more")
            lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def compare(args):
    """Compare one fast-lane run against the full run beside it."""
    report = compare_runs(results(args.fast_lane), results(args.full),
                          read_lines(args.selection) if args.selection else [])
    report.update({
        "schema": SCHEMA,
        "week": args.week,
        "commit": args.commit,
        "fast_lane_seconds": args.fast_lane_seconds,
        "escaped_rate": escaped_rate(report["failures_full"], len(report["escaped"])),
    })

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, sort_keys=True)
            handle.write("\n")
    summary = format_compare(report)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(summary)
    print(summary, end="")
    return 0


def aggregate_reports(reports):
    """Roll many ``compare`` reports into the rate step 4 should be decided on."""
    failures = sum(r["failures_full"] for r in reports)
    escaped = sum(len(r["escaped"]) for r in reports)
    times = sorted(r["fast_lane_seconds"] for r in reports if r.get("fast_lane_seconds"))
    # Every PR that failed nothing says nothing about escape; count separately
    # so a quiet fortnight cannot read as a proven-safe fast lane.
    informative = sum(1 for r in reports if r["failures_full"])
    return {
        "schema": SCHEMA,
        "runs": len(reports),
        "runs_with_failures": informative,
        "failures_full": failures,
        "escaped": escaped,
        "unstable": sum(len(r["unstable"]) for r in reports),
        "escaped_rate": escaped_rate(failures, escaped),
        "fast_lane_seconds_median": round(statistics.median(times), 1) if times else 0.0,
        "fast_lane_seconds_max": round(max(times), 1) if times else 0.0,
    }


def format_aggregate(totals):
    """Render the aggregate as Markdown."""
    median = totals["fast_lane_seconds_median"]
    lines = [
        "### Fast-lane shadow mode — running total",
        "",
        f"- {totals['runs']} shadow runs, {totals['runs_with_failures']} of them with "
        "at least one failure",
        f"- fast lane: **{median / 60:.1f} min** median, "
        f"{totals['fast_lane_seconds_max'] / 60:.1f} min worst case",
        f"- {totals['failures_full']} failures seen by the full suite",
        f"- **{totals['escaped']} escaped the fast lane "
        f"({totals['escaped_rate'] * 100:.1f}%)**",
        f"- {totals['unstable']} unstable results (disagreed between the two runs)",
    ]
    if not totals["runs_with_failures"]:
        lines += ["", "No run has failed yet, so the escaped-regression rate is not yet "
                  "measured — a 0% here means no evidence, not a safe fast lane."]
    return "\n".join(lines) + "\n"


def aggregate(args):
    """Combine every compare report under a directory."""
    reports = []
    for root, _, files in os.walk(args.inputs):
        for name in sorted(files):
            if name.endswith(".json"):
                with open(os.path.join(root, name), encoding="utf-8") as handle:
                    report = json.load(handle)
                if report.get("schema") == SCHEMA and "failures_full" in report:
                    reports.append(report)
    if not reports:
        print(f"error: no shadow reports under {args.inputs}", file=sys.stderr)
        return 1

    totals = aggregate_reports(reports)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(totals, handle, indent=1, sort_keys=True)
            handle.write("\n")
    summary = format_aggregate(totals)
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as handle:
            handle.write(summary)
    print(summary, end="")
    return 0


def main(argv=None):
    """Dispatch to compare or aggregate."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("compare", help="one fast-lane run against the full run beside it")
    one.add_argument("--fast-lane", required=True, help="JUnit from the fast-lane run")
    one.add_argument("--full", required=True, help="JUnit from the full run")
    one.add_argument("--selection", help="the week's selected test list")
    one.add_argument("--fast-lane-seconds", type=float, default=0.0, help="measured wall-clock")
    one.add_argument("--week", default="", help="ISO year-week the selection came from")
    one.add_argument("--commit", default="", help="commit under test")
    one.add_argument("--json", help="machine-readable report to write")
    one.add_argument("--summary", help="Markdown to append (e.g. $GITHUB_STEP_SUMMARY)")
    one.set_defaults(func=compare)

    many = sub.add_parser("aggregate", help="many compare reports -> the escaped rate")
    many.add_argument("--inputs", required=True, help="directory of compare --json outputs")
    many.add_argument("--json", help="machine-readable totals to write")
    many.add_argument("--summary", help="Markdown to write")
    many.set_defaults(func=aggregate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
