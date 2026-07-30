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
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

REGRESSION = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "regression"
)
sys.path.insert(0, os.path.abspath(REGRESSION))

# pylint: disable=import-error,wrong-import-position
from testing_tool import TestCase  # type: ignore

VERDICT = re.compile(r"^VERIFICATION (SUCCESSFUL|FAILED|UNKNOWN)$", re.MULTILINE)
CLAIM_HEAD = re.compile(r"^Claim (\d+):$")
CLAIM_LOC = re.compile(r"^\s+(file .* line \d+ column \d+ function .*)$")
PER_CLAIM = re.compile(r"^[✓✗] (PASSED|FAILED): '(.*)'$", re.MULTILINE)
LOCATION = re.compile(r"file .* line \d+ column \d+ function \S+")

TIMEOUT = "timeout"
NO_VERDICT = "no-verdict"

# Options that already select or reshape the claim set; re-driving them would not
# be the comparison this oracle intends.
CONFLICTING = ("--claim", "--k-induction", "--incremental-bmc", "--falsification")


def run(esbmc, args, cwd, timeout):
    try:
        proc = subprocess.run(
            [esbmc] + args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    return proc.stdout.decode("utf-8", "replace")


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


def whole_run_verdict(text):
    if text is None:
        return TIMEOUT
    found = VERDICT.findall(text)
    return found[-1] if found else NO_VERDICT


def compare(case, esbmc, timeout, max_claims):
    # Strip --multi-property from the test's own flags: passing it twice makes
    # ESBMC produce no per-claim report at all, and leaving it on the --claim
    # runs would not be a single-property run.
    base = [
        a for a in case.generate_run_argument_list(esbmc)[1:] if a != "--multi-property"
    ]
    work = tempfile.mkdtemp(prefix="oracle-claim-")
    try:
        listing = run(esbmc, ["--show-claims"] + base, work, timeout)
        if listing is None:
            return case.name, "skip", "claim listing timed out", []
        claims = parse_claims(listing)
        if len(claims) < 2:
            return case.name, "skip", f"{len(claims)} claim(s)", []
        if len(claims) > max_claims:
            return case.name, "skip", f"{len(claims)} claims over cap", []
        multi = run(esbmc, ["--multi-property"] + base, work, timeout)
        if multi is None:
            return case.name, "skip", "multi-property timed out", []

        per_claim = PER_CLAIM.findall(multi)
        # An empty report is unparseable, not "everything passed": treating it as
        # the latter would make every broken invocation look like agreement.
        if not per_claim:
            return case.name, "skip", "no per-claim report", []
        multi_by_location = {}
        for status, key in per_claim:
            location = location_of(key)
            if location:
                multi_by_location.setdefault(location, []).append(status)

        individual_by_location = {}
        for index, location in claims:
            single = whole_run_verdict(
                run(esbmc, ["--claim", str(index)] + base, work, timeout)
            )
            if single in (TIMEOUT, NO_VERDICT):
                # Says nothing about this claim; drop the whole location rather
                # than compare a short multiset against a full one.
                individual_by_location.pop(location, None)
                multi_by_location.pop(location, None)
                continue
            status = "FAILED" if single == "FAILED" else "PASSED"
            individual_by_location.setdefault(location, []).append(status)

        mismatches, ambiguous = [], 0
        for location, statuses in sorted(individual_by_location.items()):
            reported = multi_by_location.get(location, [])
            # The two interfaces do not enumerate the same property set (see the
            # module docstring), so compare only where each side has exactly one
            # entry for the location; elsewhere the pairing is a guess.
            if len(statuses) != 1 or len(reported) != 1:
                ambiguous += 1
                continue
            if statuses[0] != reported[0]:
                mismatches.append((location, statuses[0], reported[0]))

        # Aggregate check, robust to the enumeration difference and the one that
        # actually catches R19: if any claim fails on its own, the multi-property
        # run must report at least one failure.
        any_individual_failed = any(
            "FAILED" in v for v in individual_by_location.values()
        )
        any_multi_failed = any("FAILED" in v for v in multi_by_location.values())
        if any_individual_failed != any_multi_failed:
            mismatches.append(
                (
                    "<aggregate>",
                    "some claim FAILED" if any_individual_failed else "all PASSED",
                    "some claim FAILED" if any_multi_failed else "all PASSED",
                )
            )
        note = f"{len(claims)} claims, {ambiguous} location(s) not comparable"
        return case.name, "compared", note, mismatches
    finally:
        shutil.rmtree(work, ignore_errors=True)


def collect(suite, modes):
    suite = os.path.abspath(suite)
    tests = []
    for entry in sorted(os.listdir(suite)):
        directory = os.path.join(suite, entry)
        if not os.path.isfile(os.path.join(directory, "test.desc")):
            continue
        case = TestCase(directory, entry)
        if case.test_mode not in modes:
            continue
        if any(opt in case.test_args for opt in CONFLICTING):
            continue
        tests.append(case)
    return tests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esbmc", default="build/src/esbmc/esbmc")
    parser.add_argument("--suite", default="regression/esbmc")
    parser.add_argument("--modes", default="CORE")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--max-claims", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--baseline", help="see oracle_flag_parity.py --baseline")
    args = parser.parse_args()

    esbmc = os.path.abspath(args.esbmc)
    if not os.access(esbmc, os.X_OK):
        parser.error(f"not executable: {esbmc}")

    baseline = set()
    if args.baseline:
        with open(args.baseline, "r", encoding="utf-8") as handle:
            for line in handle:
                name = line.split("#", 1)[0].strip()
                if name:
                    baseline.add(name)

    tests = collect(args.suite, args.modes.split(","))
    if args.limit:
        tests = tests[: args.limit]
    print(f"{len(tests)} candidate tests, max-claims={args.max_claims}")

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
                    for location, expected, actual in mismatches:
                        print(
                            f"  MISMATCH {name} [{location}]: "
                            f"--claim says {expected}, --multi-property says {actual}",
                            flush=True,
                        )
            if done % 50 == 0:
                print(f"  ... {done}/{len(tests)}", flush=True)

    diverged_names = {name for name, _ in diverged}
    new = sorted(diverged_names - baseline)
    stale = sorted(baseline - diverged_names)

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
    if baseline:
        for name in stale:
            print(f"STALE-BASELINE {name}: listed as diverging but agrees now")
        for name in new:
            print(f"NEW-DIVERGENCE {name}")

    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
