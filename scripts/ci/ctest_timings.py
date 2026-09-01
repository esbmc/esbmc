#!/usr/bin/env python3
"""Turn a CTest JUnit report into the per-test timing table the fast lane sizes itself from.

Nothing else in the two-tier CI (esbmc/esbmc#6735) can be sized without measured
per-test durations, so the nightly full-suite run emits JUnit XML and this
converts it to ci/test-timings.json.

The XML is parsed incrementally and each element dropped once consumed: CTest
embeds every test's full stdout in <system-out>, so a whole-suite report is
hundreds of megabytes and must never be held in memory at once.
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

SCHEMA = 1


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_junit(path):
    """Yield (name, seconds, status) for every <testcase> in a CTest JUnit report."""
    for _, elem in ET.iterparse(path, events=("end", )):
        if elem.tag != "testcase":
            continue
        name = elem.get("name")
        if name:
            if elem.find("skipped") is not None:
                status = "skip"
            elif elem.find("failure") is not None or elem.get("status") == "fail":
                status = "fail"
            else:
                status = "run"
            yield name, round(float(elem.get("time") or 0.0), 3), status
        elem.clear()


def build_table(junit_path, previous=None, source=None):
    """Merge a run's measurements over ``previous``, newest measurement winning.

    Entries only in ``previous`` are carried across so a partial run (a shard, a
    suite re-run, a job that hit the wall) narrows the table's freshness rather
    than deleting the tests it did not touch.
    """
    tests = dict(previous.get("tests", {})) if previous else {}
    measured = _utc_now()
    seen = 0
    for name, seconds, status in parse_junit(junit_path):
        seen += 1
        # A skipped test measures the harness, not the test; keep any real
        # duration already on record instead of overwriting it with ~0.
        if status == "skip" and name in tests:
            continue
        tests[name] = {"seconds": seconds, "status": status, "measured": measured}
    return seen, {
        "schema": SCHEMA,
        "generated": measured,
        "source": source or {},
        "tests": dict(sorted(tests.items())),
    }


def main(argv=None):
    """Convert one JUnit report into the timing table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--junit", required=True, help="CTest --output-junit XML to read")
    parser.add_argument("--output", required=True, help="timing table to write")
    parser.add_argument("--merge",
                    action="store_true",
                    help="carry over entries of --output that this run did not measure")
    parser.add_argument("--commit", default="", help="commit the measurements were taken at")
    parser.add_argument("--runner", default="", help="runner label the measurements came from")
    parser.add_argument("--jobs", type=int, default=0, help="ctest -j used when measuring")
    args = parser.parse_args(argv)

    previous = None
    if args.merge:
        try:
            with open(args.output, encoding="utf-8") as handle:
                previous = json.load(handle)
        except FileNotFoundError:
            pass

    source = {"commit": args.commit, "runner": args.runner, "jobs": args.jobs}
    seen, table = build_table(args.junit, previous, {k: v for k, v in source.items() if v})

    if not seen:
        print(f"error: no <testcase> elements in {args.junit}", file=sys.stderr)
        return 1

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(table, handle, indent=1, sort_keys=True)
        handle.write("\n")

    total = sum(t["seconds"] for t in table["tests"].values())
    print(f"{seen} measured, {len(table['tests'])} on record, "
          f"{total / 3600:.2f} CPU-hours total -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
