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


def measured_costs(tests_table):
    """Extract known per-test costs from a timings table, skips aside.

    A test recorded only as "skip" has never actually run -- its near-zero
    duration is the harness, not the test -- so it must not be priced as a
    known cost until a real run measures it.
    """
    return {n: t["seconds"] for n, t in tests_table.items() if t.get("status") != "skip"}


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


def _seed_always_run(costs, always_run):
    """Reserve the always-run core set's cost against the budget.

    Returns ``(taken, spent)``.
    """
    taken = set()
    spent = 0.0
    for name in sorted(always_run):
        if name in costs:
            taken.add(name)
            spent += costs[name]
    return taken, spent


def _stratify(universe, taken):
    """Group the not-yet-taken tests of ``universe`` by stratum."""
    strata = {}
    for name in universe:
        if name not in taken:
            strata.setdefault(stratum_of(name), []).append(name)
    return strata


def _seed_one_per_stratum(strata, taken, costs, spent, week):
    """Guarantee every non-empty stratum at least one test.

    Keeps a small suite from going untested for weeks at a time even when its
    proportional budget share would otherwise round to zero. Returns the
    updated ``spent``.
    """
    for stratum in sorted(strata):
        rng = random.Random(seed_for(week, stratum))
        pool = [n for n in strata[stratum] if n not in taken]
        if pool:
            name = rng.choice(sorted(pool))
            taken.add(name)
            spent += costs[name]
    return spent


def _spend_stratum_share(names, share, costs, taken, round_no, week, stratum):
    # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    """Draw from one stratum's runtime terciles, in proportion to their cost.

    Returns ``(picked, spent)``.
    """
    buckets = _terciles(names, costs)
    bucket_total = sum(costs[n] for n in names)
    # A fresh RNG per (week, stratum, round) keeps the draw reproducible while
    # letting later rounds reach tests the first pass skipped.
    rng = random.Random(seed_for(week, f"{stratum}#{round_no}"))

    picked, spent = [], 0.0
    for bucket in buckets:
        weight = sum(costs[n] for n in bucket)
        bucket_picked, cost = _draw(bucket, share * weight / bucket_total, costs, rng, taken)
        picked += bucket_picked
        spent += cost
    return picked, spent


def _redistribute_round(strata, taken, costs, remaining, round_no, week):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Spend one round of ``remaining`` budget across strata.

    Returns ``(added, spent)``, where ``added`` says whether any test was
    picked this round.
    """
    pending = {s: [n for n in names if n not in taken] for s, names in strata.items()}
    pending = {s: names for s, names in pending.items() if names}
    total = sum(costs[n] for names in pending.values() for n in names)
    if not total:
        return False, 0.0

    added = False
    spent = 0.0
    for stratum in sorted(pending):
        names = pending[stratum]
        share = remaining * sum(costs[n] for n in names) / total
        picked, stratum_spent = _spend_stratum_share(names, share, costs, taken, round_no, week,
                                                     stratum)
        if picked:
            added = True
            taken.update(picked)
            spent += stratum_spent
    return added, spent


def _redistribute(strata, taken, costs, budget, spent, week):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Round-robin unspent budget across strata until nothing more fits."""
    for round_no in range(REDISTRIBUTION_ROUNDS):
        remaining = budget - spent
        if remaining <= 0:
            break
        added, round_spent = _redistribute_round(strata, taken, costs, remaining, round_no, week)
        spent += round_spent
        if not added:
            break
    return spent


def _build_stats(universe, taken, always_run, costs, spent, budget, week):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Summarize the selection over the whole universe.

    Not just the sampled part, so a suite entirely absorbed by the core set
    still shows up in the table.
    """
    everything = {}
    for name in universe:
        everything.setdefault(stratum_of(name), []).append(name)

    return {
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


def select(universe, costs, always_run, budget, week):
    """Choose the week's subset. Returns ``(sorted names, stats)``."""
    taken, spent = _seed_always_run(costs, always_run)
    strata = _stratify(universe, taken)
    spent = _seed_one_per_stratum(strata, taken, costs, spent, week)
    spent = _redistribute(strata, taken, costs, budget, spent, week)
    stats = _build_stats(universe, taken, always_run, costs, spent, budget, week)
    return sorted(taken), stats


def format_summary(stats, jobs, packing_efficiency):
    """Render the selection as Markdown for the GitHub Actions job summary."""
    wall = stats["cpu_seconds"] / max(jobs, 1) / packing_efficiency
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
    lines.append(f"Reproduce: `scripts/ci/run_selected_tests.py "
                 f"--tests ci/selected-tests-{stats['week']}.txt`")
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
    tests_table = table.get("tests", {})
    measured = measured_costs(tests_table)

    universe = read_lines(args.tests) if args.tests else sorted(tests_table)
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
        # This file, not a re-run, is the week's source of truth:
        # ci/test-timings.json refreshes nightly, so select_tests.py --week
        # can pick a different set later in the same week.
        handle.write("# reproduce: scripts/ci/run_selected_tests.py --tests <this file>\n")
        handle.write("".join(f"{name}\n" for name in selected))

    summary = format_summary(stats, args.jobs, args.packing_efficiency)
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
