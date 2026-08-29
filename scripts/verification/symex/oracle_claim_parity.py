#!/usr/bin/env python3
"""Tier-C oracle: --multi-property must agree with the per-claim runs.

H-C7 of docs/roadmap/goto-symex-verification-plan.md (§7.4). Verifying every
property in one pass and verifying each property on its own are two routes to the
same set of verdicts, so any per-claim disagreement is a real defect in one of
them. This is finer-grained than the flag-parity legs: those compare one verdict
per run, so a pair of compensating claim-level errors inside a single run cancels
out. It is also where M5 parked H-B3's per-claim residue, since a whole-run
slicing comparison has the same blind spot.

R19 (§15 M7) is exactly this shape and was found by accident, by the
--smt-during-symex leg, which is the argument for running the comparison
deliberately.

Two properties of ESBMC's interfaces shape this oracle, and both cost a wrong
answer before being accounted for.

`--claim N` does **not** verify claim N in isolation: the memory-safety checks
stay in the formula, so a FAILED run may have violated something else. On
`github_192` claims 1, 3 and 5 all report FAILED on one unrelated `dereference
failure`. A FAILED therefore counts for a claim only when the violated property
sits at that claim's own location (see `claim_verdict`); otherwise the run is
treated as saying nothing. Without that check the oracle reports a divergence for
every claim in such a test.

Claims are matched by **source location**, and the two sides are compared as a
multiset of statuses per location. Matching on the printed comment does not work:
--show-claims puts the comment and the claim expression on separate lines while
--multi-property concatenates them ("assertion" versus "assertion (_Bool)0"), and
--multi-property additionally reports properties --show-claims never lists, such
as unwinding assertions. Keying on the comment reports those differences as
missing claims, which is a bug in the oracle and not in ESBMC. Locations present
only in the --multi-property report are therefore counted, not compared.

Usage:
    oracle_claim_parity.py --esbmc build/src/esbmc/esbmc --max-claims 12
"""

import argparse
import os
import re
import sys
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

from oracle_common import (
    NO_VERDICT,
    TIMEOUT,
    capture,
    collect_tests,
    drop_scratch,
    esbmc_path,
    load_baseline,
    report_baseline,
    scratch_dir,
    verdict_of,
)

CLAIM_HEAD = re.compile(r"^Claim (\d+):$")
CLAIM_LOC = re.compile(r"^\s+(file .* line \d+ column \d+ function .*)$")
PER_CLAIM = re.compile(r"^[✓✗] (PASSED|FAILED): '(.*)'$", re.MULTILINE)
LOCATION = re.compile(r"file .* line \d+ column \d+ function \S+")

# Options that already select or reshape the claim set; re-driving them would not
# be the comparison this oracle intends.
CONFLICTING = ("--claim", "--k-induction", "--incremental-bmc", "--falsification")


def parse_claims(text):
    """[(index, location)] from --show-claims."""
    claims = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        head = CLAIM_HEAD.match(line)
        if not head:
            continue
        for follow in lines[i + 1 : i + 3]:
            loc = CLAIM_LOC.match(follow)
            if loc:
                claims.append((int(head.group(1)), loc.group(1)))
                break
    return claims


def location_of(key):
    found = LOCATION.search(key)
    return found.group(0) if found else None


VIOLATED = re.compile(r"^Violated property:\n\s+(file .* line \d+ column \d+ function \S+)",
                      re.MULTILINE)


#: One test's invocation context, threaded through the per-claim runs.
Ctx = namedtuple("Ctx", "esbmc base work timeout")


def claim_verdict(ctx, index, location):
    """What a `--claim index` run says about *that* claim.

    `--claim N` does not verify claim N alone: the memory-safety checks stay in
    the formula, so a FAILED run may have violated something else entirely. On
    `github_192` every one of claims 1, 3 and 5 reports FAILED on the same
    unrelated `dereference failure` at line 4. Taking the run verdict as the
    claim's verdict manufactures disagreements. So a FAILED counts only when the
    violated property is at the claim's own location; otherwise this run says
    nothing about it.
    """
    out = capture(ctx.esbmc, ["--claim", str(index)] + ctx.base, ctx.work, ctx.timeout)
    verdict = verdict_of(out)
    if verdict in (TIMEOUT, NO_VERDICT):
        return None
    if verdict != "FAILED":
        return "PASSED"
    violated = VIOLATED.search(out)
    if violated and violated.group(1) == location:
        return "FAILED"
    return None


def individual_statuses(ctx, claims):
    """location -> [status], keeping only claims this run can speak for."""
    by_location = {}
    inconclusive = set()
    for index, location in claims:
        status = claim_verdict(ctx, index, location)
        if status is None:
            inconclusive.add(location)
            continue
        by_location.setdefault(location, []).append(status)
    for location in inconclusive:
        by_location.pop(location, None)
    return by_location, inconclusive


def multi_statuses(text):
    by_location = {}
    for status, key in PER_CLAIM.findall(text):
        location = location_of(key)
        if location:
            by_location.setdefault(location, []).append(status)
    return by_location


def per_location_mismatches(individual, multi):
    """Compare only where each side has exactly one entry for a location: the two
    interfaces do not enumerate the same property set, so any other pairing is a
    guess. Returns (mismatches, not_comparable_count)."""
    mismatches, ambiguous = [], 0
    for location, statuses in sorted(individual.items()):
        reported = multi.get(location, [])
        if len(statuses) != 1 or len(reported) != 1:
            ambiguous += 1
        elif statuses[0] != reported[0]:
            mismatches.append((location, statuses[0], reported[0]))
    return mismatches, ambiguous


def aggregate_mismatch(individual, multi, claim_locations, complete):
    """If a claim fails on its own, the multi-property run must report a failure.

    Sound only under two conditions, both of which cost a false alarm when
    ignored. `complete` must hold: an inconclusive claim leaves the individual
    side looking failure-free when it simply has no evidence. And multi's
    failures must lie at enumerated claim locations -- it also reports
    memory-safety checks and unwinding assertions that --show-claims never
    lists, and the individual side can never match those.
    """
    if not complete:
        return None
    theirs_at_claims = [
        loc for loc, v in multi.items() if "FAILED" in v and loc in claim_locations
    ]
    if any("FAILED" in v for v in multi.values()) and not theirs_at_claims:
        return None
    mine = any("FAILED" in v for v in individual.values())
    theirs = bool(theirs_at_claims)
    if mine == theirs:
        return None
    label = {True: "some claim FAILED", False: "all PASSED"}
    return ("<aggregate>", label[mine], label[theirs])


def eligible_claims(ctx, max_claims):
    """The test's claims, or a skip reason. Fewer than two claims makes the
    comparison vacuous, and the cap bounds cost: a test with 38 claims is 40
    ESBMC runs."""
    listing = capture(ctx.esbmc, ["--show-claims"] + ctx.base, ctx.work, ctx.timeout)
    if listing is None:
        return None, "claim listing timed out"
    claims = parse_claims(listing)
    if len(claims) < 2:
        return None, f"{len(claims)} claim(s)"
    if len(claims) > max_claims:
        return None, f"{len(claims)} claims over cap"
    return claims, None


def base_flags(case, esbmc):
    # Strip --multi-property from the test's own flags: passing it twice makes
    # ESBMC produce no per-claim report at all, and leaving it on the --claim
    # runs would not be a single-property run.
    return [
        a for a in case.generate_run_argument_list(esbmc)[1:] if a != "--multi-property"
    ]


def evaluate(ctx, claims):
    """Diff the two sides for one test: (mismatches, note), or (None, reason)
    when the run cannot support a comparison."""
    listing = capture(ctx.esbmc, ["--multi-property"] + ctx.base, ctx.work, ctx.timeout)
    if listing is None:
        return None, "multi-property timed out"
    multi = multi_statuses(listing)
    # An empty report is unparseable, not "everything passed": treating it as the
    # latter would make every broken invocation look like agreement.
    if not multi:
        return None, "no per-claim report"

    individual, unspeakable = individual_statuses(ctx, claims)
    for location in unspeakable:
        multi.pop(location, None)

    mismatches, ambiguous = per_location_mismatches(individual, multi)
    aggregate = aggregate_mismatch(
        individual, multi, {loc for _, loc in claims}, not unspeakable
    )
    if aggregate:
        mismatches.append(aggregate)
    note = (
        f"{len(claims)} claims, {ambiguous} location(s) not comparable, "
        f"{len(unspeakable)} inconclusive"
    )
    return mismatches, note


def compare(case, esbmc, timeout, max_claims):
    work = scratch_dir("oracle-claim-")
    ctx = Ctx(esbmc, base_flags(case, esbmc), work, timeout)
    try:
        claims, reason = eligible_claims(ctx, max_claims)
        if reason:
            return case.name, "skip", reason, []
        mismatches, note = evaluate(ctx, claims)
        if mismatches is None:
            return case.name, "skip", note, []
        return case.name, "compared", note, mismatches
    finally:
        drop_scratch(work)


def announce(name, mismatches):
    for location, expected, actual in mismatches:
        print(
            f"  MISMATCH {name} [{location}]: "
            f"--claim says {expected}, --multi-property says {actual}",
            flush=True,
        )


def sweep(tests, esbmc, args):
    """Run every test through compare(), reporting as results land.

    Returns (compared_count, [(name, skip_reason)], [(name, mismatches)]).
    """
    compared, skipped, diverged = 0, [], []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(compare, t, esbmc, args.timeout, args.max_claims)
            for t in tests
        ]
        for done, future in enumerate(futures, 1):
            name, status, note, mismatches = future.result()
            if status == "skip":
                skipped.append((name, note))
            else:
                compared += 1
                if mismatches:
                    diverged.append((name, mismatches))
                    announce(name, mismatches)
            if done % 50 == 0:
                print(f"  ... {done}/{len(tests)}", flush=True)
    return compared, skipped, diverged


def report(compared, diverged, skipped):
    """Counts, then every mismatch by name and the skip reasons by frequency."""
    print(f"\ncompared     {compared}  (>= 2 distinct claims, within the cap)")
    print(f"diverged     {len(diverged)}")
    print(f"skipped      {len(skipped)}")
    for name, mismatches in diverged:
        for location, expected, actual in mismatches:
            print(f"MISMATCH {name} [{location}]: {expected} vs {actual}")
    reasons = {}
    for _, note in skipped:
        reasons[note] = reasons.get(note, 0) + 1
    for note, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"SKIPPED {count}: {note}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esbmc", default="build/src/esbmc/esbmc")
    parser.add_argument("--suite", default="regression/esbmc")
    parser.add_argument("--modes", default="CORE")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--max-claims", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--baseline", help="see oracle_flag_parity.py --baseline")
    return parser



def main():
    parser = build_parser()
    args = parser.parse_args()
    esbmc = esbmc_path(parser, args.esbmc)
    baseline = load_baseline(args.baseline)
    tests = collect_tests(args.suite, args.modes.split(","), CONFLICTING)
    if args.limit:
        tests = tests[: args.limit]
    print(f"{len(tests)} candidate tests, max-claims={args.max_claims}")

    compared, skipped, diverged = sweep(tests, esbmc, args)
    report(compared, diverged, skipped)
    return 1 if report_baseline(baseline, {n for n, _ in diverged}) else 0


if __name__ == "__main__":
    sys.exit(main())
