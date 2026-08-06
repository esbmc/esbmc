#!/usr/bin/env python3
"""Self-test for ccn_report.py. Run: python3 scripts/complexity/test_ccn_report.py

Each case is built from the fixture pair in testdata/{before,after}, which is a
miniature of the repo layout (src/, regression/) so partitioning is exercised
too rather than stubbed.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import ccn_report as ccn  # noqa: E402

DATA = HERE / "testdata"
FAILURES = []


def check(condition, message):
    """Record `message` when `condition` does not hold."""
    if not condition:
        FAILURES.append(message)


def test_partitioning():
    """Every path lands in the partition its gate depends on."""
    cases = {
        "src/goto-symex/symex_main.cpp": "core",
        "src/util/irep/migrate.h": "core",
        "src/c2goto/library/string.c": "om",
        "src/cpp/library/vector": None,
        "src/python-frontend/models/esbmc.py": "python",
        "src/python-frontend/libs/ast2json/ast2json.py": None,
        "src/c2goto/library/libm/musl/exp.c": None,
        "src/ansi-c/cpp/cpp.cpp": None,
        "regression/esbmc/github_1/main.c": "tests",
        "unit/goto-symex/renaming.test.cpp": "tests",
        "docs/roadmap/plan.md": None,
        "scripts/complexity/ccn_report.py": "python",
    }
    for path, expected in cases.items():
        got = ccn.partition_for(path)
        check(got == expected, f"partition_for({path}) = {got}, want {expected}")


def test_modified_ccn_collapses_switch():
    """The `modified` extension must collapse a wide switch to one branch."""
    key = ("core", "src/dispatch.cpp", "dispatch( int k)")
    lax = ccn.collect_files(DATA / "before", ["src/dispatch.cpp"])
    strict = ccn.collect_files(DATA / "before", ["src/dispatch.cpp"], strict=True)
    check(key in lax and key in strict, f"dispatch not found: {sorted(lax)}")
    if key in lax and key in strict:
        check(
            lax[key]["ccn"] < strict[key]["ccn"] // 2,
            f"switch not collapsed: modified {lax[key]['ccn']} vs "
            f"strict {strict[key]['ccn']}",
        )


# What changed_tests() would return for this fixture pair: the advisory files
# are analysed by name, exactly as production does, not by walking regression/.
TOUCHED_TESTS = ["regression/esbmc/witness/main.c"]


def test_rules():
    """R1/R2 fire on the fixture pair, and only where they should."""
    base = ccn.analyse(DATA / "before", TOUCHED_TESTS, threads=1)
    head = ccn.analyse(DATA / "after", TOUCHED_TESTS, threads=1)
    violations, advisory, budget = ccn.compare(base, head)
    check(
        any(a["name"] == "nasty_reduced_witness" for a in advisory),
        "the touched regression file should still reach the advisory list",
    )
    flagged = {(v["name"], v["partition"]) for v in violations}

    check(
        ("worsened", "core") in flagged,
        "R1 missed a function whose CCN rose above the threshold",
    )
    check(
        ("added", "core") in flagged,
        "R1 missed a newly added over-threshold function",
    )
    check(
        ("dispatch", "core") not in flagged,
        "an untouched over-threshold dispatcher must not be flagged",
    )
    check(
        ("nasty_reduced_witness", "tests") not in flagged,
        "regression/ is advisory: it must never produce a violation",
    )
    check(
        any(a["name"] == "grew_a_little" for a in advisory),
        "a rising but below-threshold function belongs in the advisory list",
    )
    check(
        not any(a["name"] == "tiny" for a in advisory),
        f"a rise below ADVISORY_FLOOR ({ccn.ADVISORY_FLOOR}) is noise, not signal",
    )
    check(
        ccn.budget_regressions(budget).get("core", {}).get("after", 0) > 0,
        f"R2 should record a core budget increase, got {budget}",
    )
    # `om` is looser than `core`: the same CCN must not trip it.
    check(
        ("model_strcmp", "om") not in flagged,
        "om threshold (25) should tolerate a CCN that core (15) rejects",
    )
    # Moved file + new parameter + lower CCN: neither the path nor the
    # long_name survives, so only the name fallback keeps this off the gate.
    check(
        ("relocated", "core") not in flagged,
        "a function that moved and lost complexity must not read as new",
    )


def test_clean_pr_is_silent():
    """A PR that changes nothing produces no violation and says so."""
    base = ccn.analyse(DATA / "before", TOUCHED_TESTS, threads=1)
    violations, _, budget = ccn.compare(base, base)
    check(not violations, "comparing a tree with itself must yield no violations")
    check(not ccn.budget_regressions(budget), "self-comparison must not ratchet")
    check(
        "No function this PR adds or worsens" in ccn.render(violations, [], budget),
        "the clean report should say so explicitly",
    )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    for failure in FAILURES:
        print(f"FAIL: {failure}", file=sys.stderr)
    print(f"{len(FAILURES)} failure(s)")
    sys.exit(1 if FAILURES else 0)
