#!/usr/bin/env python3
"""Per-test durations from a CTest log, and the shard layout they imply.

Reads the `N/M Test #i: name ... Passed  X sec` lines a CTest run prints,
splits the tests into the Python and native families of ci_selection,
stripes each family as `ctest -I i,,k` does, and simulates CTest's parallel
slots with a greedy scheduler in index order. Simulated minutes are an
estimate, not a CI measurement.
"""

import argparse
import heapq
import re
import sys

import ci_selection

# CTest pads the name with dots, except when the name fills the column; a
# Catch2 name may start with a space.
_LINE = re.compile(r"\d+/\d+ Test +#(?P<index>\d+): (?P<name>.+?)(?: \.+)?\s*(?:\*{3})?"
                   r"(?:Passed|Failed|Skipped|Not Run|Timeout|Exception: \w+)"
                   r"\s+(?P<seconds>[\d.]+) sec\s*$")


def read_durations(lines):
    """(index, name, seconds) of every finished test, in CTest index order."""
    found = {}
    for line in lines:
        match = _LINE.search(line)
        if match:
            found[int(match["index"])] = (match["name"].rstrip(), float(match["seconds"]))
    return [(index, *found[index]) for index in sorted(found)]


def family(name):
    """The executor family of a test, from its regression suite."""
    if not name.startswith("regression/"):
        return "producer"
    suite = name.split("/")[1] + "/"
    if suite == ci_selection.INTENSIVE_LABEL:
        return "excluded"
    return "python" if suite in ci_selection.PYTHON_LABELS else "native"


def makespan(durations, slots):
    """Seconds CTest needs for `durations`, started in order on `slots` slots."""
    finish = [0.0] * slots
    for seconds in durations:
        heapq.heapreplace(finish, finish[0] + seconds)
    return max(finish)


def shard_seconds(tests, shard_counts, slots):
    """executor -> simulated seconds, striping each family as `-I i,,k` does."""
    result = {}
    for name, count in shard_counts:
        durations = [seconds for _, test, seconds in tests if family(test) == name]
        for i in range(count):
            result[f"{name}-{i + 1}"] = makespan(durations[i::count], slots)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="CTest output, such as a GitHub job log")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--python-shards",
                        type=int,
                        default=dict(ci_selection.SHARD_COUNTS)["python"])
    parser.add_argument("--native-shards",
                        type=int,
                        default=dict(ci_selection.SHARD_COUNTS)["native"])
    args = parser.parse_args(argv)
    with open(args.log, encoding="utf-8", errors="replace") as fp:
        tests = read_durations(fp)
    total = sum(seconds for _, _, seconds in tests)
    print(f"{len(tests)} tests, {total:.2f}s of test time; monolithic -j{args.jobs} simulated "
          f"{makespan([s for _, _, s in tests], args.jobs):.2f}s")
    layout = (("python", args.python_shards), ("native", args.native_shards))
    for executor, seconds in shard_seconds(tests, layout, args.jobs).items():
        print(f"{executor}: {seconds / 60:.2f} min simulated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
