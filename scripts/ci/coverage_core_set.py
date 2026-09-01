#!/usr/bin/env python3
"""Derive the fast lane's always-run core set from per-test line coverage.

Randomly sampling the suite leaves gaps every week, so a small set of tests runs
unconditionally on every PR. Rather than hand-picking it, this builds it the way
esbmc/esbmc#6735 asks: take each test's own coverage and greedily solve a
weighted set cover for the most lines reached per second of runtime.

Two steps, because the measuring is expensive and the choosing is not:

``collect``
    Turn a tree of per-test raw profiles (ENABLE_PER_TEST_COVERAGE) into one
    covered-line bitset per test. This is ~1 llvm-cov export per test over a
    large binary, so it is a monthly self-hosted job, parallel over --jobs.

``select``
    Greedy cost-aware cover over those bitsets, emitting ci/core-set.txt. Cheap
    enough to re-run with different budgets against a single collect.

Line sets are carried as Python ints used as bitsets: 180k source lines is a
22 kB int, so the whole suite is a few hundred MB in memory and each round of
the greedy loop is a native AND plus popcount per candidate.
"""

import argparse
import base64
import gzip
import json
import os
import subprocess
import sys
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor

# Paths that are not ESBMC's own code and would distort the cover.
DEFAULT_EXCLUDES = ("/usr/", "/build/")

LINE_INDEX = "lines.txt.gz"
BITSETS = "per-test-lines.jsonl.gz"


def unmangle(dirname):
    """Recover a ctest name from the profile directory ENABLE_PER_TEST_COVERAGE made."""
    return dirname.replace("@", "/")


def profile_dirs(root):
    """Yield ``(test name, directory)`` for every per-test profile directory."""
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            yield unmangle(name), path


def _lcov_lines(binary, profdata, llvm_cov, excludes):
    """Return ``(all lines, covered lines)`` as ``file:line`` strings from an lcov export."""
    out = subprocess.run(
        [llvm_cov, "export", "-format=lcov", f"-instr-profile={profdata}", binary],
        check=True,
        capture_output=True,
        text=True).stdout
    every, covered = [], []
    source = None
    for line in out.splitlines():
        if line.startswith("SF:"):
            path = line[3:]
            source = None if any(x in path for x in excludes) else path
        elif line.startswith("DA:") and source:
            number, _, count = line[3:].partition(",")
            key = f"{source}:{number}"
            every.append(key)
            if count.strip() not in ("0", ""):
                covered.append(key)
    return every, covered


def _merge(profraws, llvm_profdata, out):
    subprocess.run([llvm_profdata, "merge", "-sparse", "-o", out] + profraws,
                   check=True,
                   capture_output=True)


def _collect_one(job):  # pylint: disable=too-many-locals
    """Worker: merge one test's profiles and return its covered lines as a bitset."""
    name, directory, binary, llvm_cov, llvm_profdata, excludes, index = job
    profraws = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".profraw")
    ]
    if not profraws:
        return name, None
    with tempfile.TemporaryDirectory() as tmp:
        profdata = os.path.join(tmp, "test.profdata")
        try:
            _merge(profraws, llvm_profdata, profdata)
            _, covered = _lcov_lines(binary, profdata, llvm_cov, excludes)
        except subprocess.CalledProcessError:
            # A test that crashed the profile writer leaves a truncated raw
            # profile. Drop it rather than failing the whole collection.
            return name, None
    bits = 0
    for key in covered:
        position = index.get(key)
        if position is not None:
            bits |= 1 << position
    if not bits:
        return name, None
    length = (bits.bit_length() + 7) // 8
    return name, base64.b64encode(zlib.compress(bits.to_bytes(length, "big"))).decode("ascii")


def collect(args):  # pylint: disable=too-many-locals
    """Build the global line index, then one covered-line bitset per test."""
    tests = list(profile_dirs(args.profiles))
    if not tests:
        print(f"error: no per-test profile directories under {args.profiles}", file=sys.stderr)
        return 1

    excludes = tuple(args.exclude) if args.exclude else DEFAULT_EXCLUDES
    os.makedirs(args.output, exist_ok=True)

    # One export over everything merged fixes the bit positions, so every
    # worker can map a line to the same bit without coordinating.
    all_profraws = [
        os.path.join(d, f) for _, d in tests for f in os.listdir(d) if f.endswith(".profraw")
    ]
    if not all_profraws:
        print(f"error: no .profraw files under {args.profiles}", file=sys.stderr)
        return 1
    print(f"indexing lines from {len(all_profraws)} raw profiles...", file=sys.stderr)
    with tempfile.TemporaryDirectory() as tmp:
        merged = os.path.join(tmp, "all.profdata")
        _merge(all_profraws, args.llvm_profdata, merged)
        every, _ = _lcov_lines(args.binary, merged, args.llvm_cov, excludes)
    index = {key: i for i, key in enumerate(dict.fromkeys(every))}
    print(f"{len(index)} instrumented lines, {len(tests)} tests", file=sys.stderr)

    with gzip.open(os.path.join(args.output, LINE_INDEX), "wt", encoding="utf-8") as handle:
        handle.write("".join(f"{key}\n" for key in index))

    jobs = [(name, d, args.binary, args.llvm_cov, args.llvm_profdata, excludes, index)
            for name, d in tests]
    done = 0
    with gzip.open(os.path.join(args.output, BITSETS), "wt", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for name, bits in pool.map(_collect_one, jobs, chunksize=4):
                done += 1
                if bits:
                    handle.write(json.dumps({"test": name, "bits": bits}) + "\n")
                if done % 250 == 0:
                    print(f"  {done}/{len(tests)}", file=sys.stderr)
    print(f"wrote {args.output}/{BITSETS}", file=sys.stderr)
    return 0


def load_bitsets(path):
    """Read the collected per-test bitsets back as ``{test: int}``."""
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            raw = zlib.decompress(base64.b64decode(record["bits"]))
            out[record["test"]] = int.from_bytes(raw, "big")
    return out


def greedy_cover(bitsets, costs, budget, target=1.0):  # pylint: disable=too-many-locals
    """Pick tests maximising newly covered lines per second, within ``budget``.

    Returns ``(chosen, trail)`` where ``trail`` records the cumulative fraction
    of the achievable union covered after each pick.
    """
    achievable = 0
    for bits in bitsets.values():
        achievable |= bits
    total = achievable.bit_count()
    if not total:
        return [], []

    remaining = dict(bitsets)
    covered = 0
    spent = 0.0
    chosen, trail = [], []

    while remaining and covered.bit_count() < total * target:
        best, best_value, best_gain = None, 0.0, 0
        for name, bits in remaining.items():
            cost = costs.get(name, 1.0)
            if spent + cost > budget:
                continue
            gain = (bits & ~covered).bit_count()
            # Zero-cost tests would divide by zero and are not free anyway;
            # floor the divisor at a millisecond.
            value = gain / max(cost, 0.001)
            if value > best_value:
                best, best_value, best_gain = name, value, gain
        if best is None or not best_gain:
            break
        covered |= remaining.pop(best)
        spent += costs.get(best, 1.0)
        chosen.append(best)
        trail.append(round(covered.bit_count() / total, 4))

    return chosen, trail


def select(args):
    """Solve the cover and write ci/core-set.txt."""
    bitsets = load_bitsets(os.path.join(args.coverage, BITSETS))
    if not bitsets:
        print(f"error: no coverage data in {args.coverage}", file=sys.stderr)
        return 1

    with open(args.timings, encoding="utf-8") as handle:
        costs = {n: t["seconds"] for n, t in json.load(handle).get("tests", {}).items()}
    missing = [n for n in bitsets if n not in costs]
    if missing:
        print(f"note: {len(missing)} covered tests have no timing; assuming 1s", file=sys.stderr)
        costs.update({n: 1.0 for n in missing})

    budget = args.budget_seconds * max(args.jobs, 1)
    chosen, trail = greedy_cover(bitsets, costs, budget, args.target)
    if not chosen:
        print("error: no test fits the budget", file=sys.stderr)
        return 1

    spent = sum(costs[n] for n in chosen)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("# Coverage-derived always-run core set (esbmc/esbmc#6735).\n")
        handle.write("# Regenerate: scripts/ci/coverage_core_set.py select\n")
        handle.write(f"# {len(chosen)} tests, {trail[-1] * 100:.1f}% of reachable lines, "
                 f"{spent:.0f} CPU-seconds\n")
        handle.write("".join(f"{name}\n" for name in sorted(chosen)))

    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write("### Coverage-derived core set\n\n")
            handle.write(f"- {len(chosen)} of {len(bitsets)} tests\n")
            handle.write(f"- {trail[-1] * 100:.1f}% of the lines the suite reaches at all\n")
            handle.write(f"- {spent / 60:.1f} CPU-minutes "
                     f"({spent / max(args.jobs, 1) / 60:.1f} min at -j{args.jobs})\n\n")
            handle.write("| # tests | cumulative coverage |\n| ---: | ---: |\n")
            for i in range(0, len(trail), max(len(trail) // 20, 1)):
                handle.write(f"| {i + 1} | {trail[i] * 100:.1f}% |\n")

    print(f"{len(chosen)} tests, {trail[-1] * 100:.1f}% of reachable lines -> {args.output}")
    return 0


def main(argv=None):
    """Dispatch to collect or select."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    gather = sub.add_parser("collect", help="per-test raw profiles -> covered-line bitsets")
    gather.add_argument("--profiles", required=True, help="the run's per-test profile tree")
    gather.add_argument("--binary", required=True, help="the instrumented esbmc binary")
    gather.add_argument("--output", required=True, help="directory to write the bitsets to")
    gather.add_argument("--jobs", type=int, default=os.cpu_count(), help="parallel llvm-cov runs")
    gather.add_argument("--exclude", action="append", help="path fragment to ignore (repeatable)")
    gather.add_argument("--llvm-cov", default="llvm-cov")
    gather.add_argument("--llvm-profdata", default="llvm-profdata")
    gather.set_defaults(func=collect)

    solve = sub.add_parser("select", help="bitsets -> ci/core-set.txt")
    solve.add_argument("--coverage", required=True, help="directory produced by collect")
    solve.add_argument("--timings", default="ci/test-timings.json", help="per-test durations")
    solve.add_argument("--output", default="ci/core-set.txt")
    solve.add_argument("--budget-seconds", type=float, default=180.0, help="wall-clock to fill")
    solve.add_argument("--jobs", type=int, default=2, help="ctest -j the budget assumes")
    solve.add_argument("--target",
                       type=float,
                       default=0.95,
                       help="stop once this fraction of reachable lines is covered")
    solve.add_argument("--report", help="Markdown report to write")
    solve.set_defaults(func=select)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
