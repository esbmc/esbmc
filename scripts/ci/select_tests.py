#!/usr/bin/env python3
"""Pick the week's fast-lane test subset (esbmc/esbmc#6735, tier 1).

The selection is a function of the ISO week alone, so every PR opened in a given
week runs exactly the same tests and a failure is reproducible locally with
``--week``. There is no per-PR roll to blame a red build on.

Two properties the sampling has to keep:

* **No area goes dark.** Sampling is stratified by suite (the ctest label) and,
  within a suite, by runtime tercile, so a whole frontend or solver backend
  cannot sit out a week, and slow tests are not squeezed out by cheap ones.
* **A suite's draw order is its own.** Each stratum shuffles from a seed derived
  from the week *and* the stratum name, so the order a suite's tests are
  considered in does not depend on what any other suite contains. (Budget shares
  are still global -- growing one suite shifts everyone's share -- but no suite
  reshuffles because another one changed.)

The budget is measured cumulative runtime, not a test count: tests are added
until the projected wall-clock hits ``--budget-seconds``.
"""

import argparse
import hashlib
import json
import random
import statistics
import sys
from datetime import date

# Wall-clock is never jobs x CPU-seconds: the tail of a run is imperfectly
# packed and the last few long tests finish alone. Discount the budget so the
# projection does not systematically overshoot.
DEFAULT_PACKING_EFFICIENCY = 0.85

# Cost assumed for a test with no measurement and no measured peers to impute
# from -- only reachable on a first run, before any timing table exists.
FALLBACK_COST = 1.0

# No test is free. A suite whose tests all measured zero -- every one of them
# skipped on the measuring host, say -- would otherwise carry zero weight and
# divide by it when its budget share is split across runtime buckets.
MIN_COST = 0.001

# Budget left unspent by exhausted strata is redistributed over the rest. A
# handful of rounds converges; the loop also stops as soon as a round adds
# nothing.
REDISTRIBUTION_ROUNDS = 8


def iso_week(today=None):
    """Return the ISO year-week label, e.g. ``2026-W36``, used as the seed."""
    year, week, _ = (today or date.today()).isocalendar()
    return f"{year}-W{week:02d}"


def seed_for(week, stratum=""):
    """Derive a stable 64-bit seed from the week and (optionally) a stratum."""
    digest = hashlib.sha256(f"{week}\0{stratum}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def stratum_of(name):
    """Map a ctest name to its stratum: the regression suite, else ``unit``.

    Regression tests are registered as ``regression/<suite>/<test>`` where
    ``<suite>`` is exactly the ctest label (regression/CMakeLists.txt), so
    stripping the leading ``regression/`` and the trailing test name recovers it.
    """
    if not name.startswith("regression/"):
        return "unit"
    rest = name[len("regression/"):]
    suite, sep, _ = rest.rpartition("/")
    return suite if sep else "unit"


def read_lines(path):
    """Read a newline-delimited list, dropping blanks and ``#`` comments."""
    with open(path, encoding="utf-8") as handle:
        return [s for s in (line.split("#", 1)[0].strip() for line in handle) if s]


def impute_costs(universe, measured):
    """Give every test a cost, standing in a peer's median where none was measured.

    A test with no timing yet -- newly added, or skipped on the measuring host --
    must not read as free, or the sampler would happily take an unbounded number
    of them.
    """
    known = {n: measured[n] for n in universe if n in measured}
    per_stratum = {}
    for name, cost in known.items():
        per_stratum.setdefault(stratum_of(name), []).append(cost)
    overall = statistics.median(known.values()) if known else FALLBACK_COST

    costs = {}
    for name in universe:
        if name in known:
            cost = known[name]
        else:
            peers = per_stratum.get(stratum_of(name))
            cost = statistics.median(peers) if peers else overall
        costs[name] = max(cost, MIN_COST)
    return costs


def _terciles(names, costs):
    """Split a stratum into fast/medium/slow buckets of roughly equal size."""
    ordered = sorted(names, key=lambda n: (costs[n], n))
    if len(ordered) < 3:
        return [ordered]
    cut = len(ordered) // 3
    return [ordered[:cut], ordered[cut:2 * cut], ordered[2 * cut:]]


def _draw(bucket, budget, costs, rng, taken):
    """Randomly take tests from ``bucket`` until ``budget`` cannot fit another.

    Returns the tests taken and what they cost. Sampling is uniform within the
    bucket rather than cheapest-first, so the choice stays unbiased.
    """
    pool = [n for n in bucket if n not in taken]
    rng.shuffle(pool)
    picked, spent = [], 0.0
    for name in pool:
        if spent + costs[name] > budget:
            continue
        picked.append(name)
        spent += costs[name]
    return picked, spent


def select(universe, costs, always_run, budget, week):
    # pylint: disable=too-many-locals,too-many-branches
    """Choose the week's subset. Returns ``(sorted names, stats)``."""
    taken = set()
    spent = 0.0
    for name in sorted(always_run):
        if name in costs:
            taken.add(name)
            spent += costs[name]

    strata = {}
    for name in universe:
        if name not in taken:
            strata.setdefault(stratum_of(name), []).append(name)

    # Every non-empty stratum contributes at least one test even if its
    # proportional share would round to zero, which is what keeps a small suite
    # from going untested for weeks at a time.
    for stratum in sorted(strata):
        rng = random.Random(seed_for(week, stratum))
        pool = [n for n in strata[stratum] if n not in taken]
        if pool:
            name = rng.choice(sorted(pool))
            taken.add(name)
            spent += costs[name]

    for round_no in range(REDISTRIBUTION_ROUNDS):
        remaining = budget - spent
        if remaining <= 0:
            break
        pending = {
            s: [n for n in names if n not in taken]
            for s, names in strata.items()
        }
        pending = {s: names for s, names in pending.items() if names}
        total = sum(costs[n] for names in pending.values() for n in names)
        if not total:
            break

        added = False
        for stratum in sorted(pending):
            names = pending[stratum]
            share = remaining * sum(costs[n] for n in names) / total
            buckets = _terciles(names, costs)
            bucket_total = sum(costs[n] for n in names)
            # A fresh RNG per (week, stratum, round) keeps the draw reproducible
            # while letting later rounds reach tests the first pass skipped.
            rng = random.Random(seed_for(week, f"{stratum}#{round_no}"))
            for bucket in buckets:
                weight = sum(costs[n] for n in bucket)
                picked, cost = _draw(bucket, share * weight / bucket_total, costs, rng, taken)
                if picked:
                    added = True
                    taken.update(picked)
                    spent += cost
        if not added:
            break

    # Report over the whole universe, not just the sampled part, so a suite
    # entirely absorbed by the core set still shows up in the table.
    everything = {}
    for name in universe:
        everything.setdefault(stratum_of(name), []).append(name)

    stats = {
        "week": week,
        "selected": len(taken),
        "universe": len(universe),
        "always_run": len(set(always_run) & set(costs)),
        "cpu_seconds": round(spent, 1),
        "budget_cpu_seconds": round(budget, 1),
        "strata": {
            s: {
                "selected": sum(1 for n in names if n in taken),
                "total": len(names)
            }
            for s, names in sorted(everything.items())
        },
    }
    return sorted(taken), stats


def format_summary(stats, jobs):
    """Render the selection as Markdown for the GitHub Actions job summary."""
    wall = stats["cpu_seconds"] / max(jobs, 1)
    lines = [
        f"### Fast-lane selection `{stats['week']}`",
        "",
        f"- {stats['selected']} of {stats['universe']} tests "
        f"({100 * stats['selected'] / max(stats['universe'], 1):.1f}%)",
        f"- {stats['always_run']} always-run (core set)",
        f"- {stats['cpu_seconds'] / 60:.1f} CPU-minutes, "
        f"projected {wall / 60:.1f} min wall-clock at -j{jobs}",
        "",
        "| Suite | Selected | Total |",
        "| --- | ---: | ---: |",
    ]
    lines += [
        f"| `{s}` | {v['selected']} | {v['total']} |" for s, v in stats["strata"].items()
    ]
    lines.append("")
    lines.append(f"Reproduce locally: `scripts/ci/select_tests.py --week {stats['week']}`")
    return "\n".join(lines) + "\n"


def main(argv=None):
    """Write the week's selection and its summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timings",
                        default="ci/test-timings.json",
                        help="measured per-test durations")
    parser.add_argument("--tests", help="newline-delimited universe; defaults to the timed tests")
    parser.add_argument("--week", default=None, help="ISO year-week seed (default: this week)")
    parser.add_argument("--budget-seconds", type=float, default=900.0, help="wall-clock budget")
    parser.add_argument("--jobs", type=int, default=2, help="ctest -j the budget assumes")
    parser.add_argument("--packing-efficiency",
                    type=float,
                    default=DEFAULT_PACKING_EFFICIENCY,
                    help="fraction of jobs x budget a real run actually packs")
    parser.add_argument("--always-run", help="core set to include unconditionally")
    parser.add_argument("--output", required=True, help="selected test list to write")
    parser.add_argument("--summary", help="Markdown summary to write")
    args = parser.parse_args(argv)

    with open(args.timings, encoding="utf-8") as handle:
        table = json.load(handle)
    measured = {n: t["seconds"] for n, t in table.get("tests", {}).items()}

    universe = read_lines(args.tests) if args.tests else sorted(measured)
    if not universe:
        print("error: empty test universe", file=sys.stderr)
        return 1

    always_run = read_lines(args.always_run) if args.always_run else []
    week = args.week or iso_week()
    costs = impute_costs(universe, measured)
    budget = args.budget_seconds * max(args.jobs, 1) * args.packing_efficiency

    selected, stats = select(universe, costs, always_run, budget, week)

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(f"# fast-lane selection for {week}\n")
        handle.write(f"# regenerate: scripts/ci/select_tests.py --week {week}\n")
        handle.write("".join(f"{name}\n" for name in selected))

    summary = format_summary(stats, args.jobs)
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as handle:
            handle.write(summary)
    print(summary, end="")

    if stats["cpu_seconds"] > budget:
        print(
            "warning: the core set plus one test per suite already exceeds the budget; "
            "the fast lane will overrun",
            file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
