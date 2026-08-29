#!/usr/bin/env python3
"""Cyclomatic-complexity report and ratchet for ESBMC pull requests.

ESBMC's median function has a CCN of 3, but ~11% sit above 15 and the top of the
list is hand-rolled AST dispatch (`migrate_expr`, `convert_ast_node`,
`clang_c_convertert::get_expr`). A repo-wide absolute gate would therefore be
permanently red and would only ever nag about jump tables nobody intends to
refactor. This script gates on what a *pull request changes* instead.

Two rules, both evaluated between the merge base and the PR head:

  R1 new-or-worsened  A function the PR adds lands above its partition's
                      threshold, or an existing function's CCN increases and
                      ends above it. Merely touching a file that already
                      contains a 300-CCN dispatcher is not a violation, and a
                      function that moves or gains a parameter is matched by
                      name rather than being read as newly introduced.
  R2 budget ratchet   The per-partition count of over-threshold functions may
                      not increase. Catches the growth R1's per-function
                      matching cannot see, such as one big function replaced
                      by several renamed ones that are each still too complex.

Complexity is *modified* CCN (lizard's `modified` extension): a switch/match
with N arms counts as one decision point, not N. Note that this cannot collapse
a hand-rolled `if (x == A) ... if (x == B) ...` chain, which is why some
dispatchers stay high; that is the metric working, not a misconfiguration.

Sources are split into four partitions so that `regression/` — 565k LOC of
deliberately pathological creduce and csmith output — is reported but never
gated. See PARTITIONS / THRESHOLDS below.

    ccn_report.py                        # working tree vs merge base with master
    ccn_report.py --base <rev> --gate    # exit 1 on any R1/R2 violation
    ccn_report.py --snapshot             # whole-tree ranking, no comparison
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

try:
    import lizard
except ImportError:  # pragma: no cover - install guidance is the whole message
    sys.exit("error: lizard is not installed (pip install lizard==1.23.0)")

# The gated partitions live entirely under these roots. `unit/` and
# `regression/` are deliberately absent: R2 never counts them, and walking 565k
# LOC of creduce output on both revisions costs minutes for a report nobody
# gates on, so the `tests` partition is analysed only on the files a PR touches.
ROOTS = ("src", "scripts")

# Vendored or third-party trees. The first three mirror the clang-format job in
# .github/workflows/pull_request.yml; ast2json is the vendored Python parser.
EXCLUDED = (
    "src/ansi-c/cpp/",
    "src/clang-c-frontend/headers/",
    "src/c2goto/library/libm/musl/",
    "src/python-frontend/libs/",
)

C_EXTS = (".c", ".cpp", ".cc", ".h", ".hpp", ".hh")
OM_PREFIXES = ("src/c2goto/library/", "src/cpp/library/")
PY_PREFIXES = ("src/python-frontend/", "scripts/")
TEST_PREFIXES = ("unit/", "regression/")

# A partition with a threshold is gated; `tests` has none and stays advisory.
# `om` is looser because the hand-written C models are branchy by design.
THRESHOLDS = {"core": 15, "python": 15, "om": 25}
PARTITIONS = ("core", "python", "om", "tests")

# Rising-but-tolerable functions are worth showing; a brand-new CCN-2 accessor
# is not. Only report the advisory tail from here up.
ADVISORY_FLOOR = 8


def partition_for(rel):
    """Return the partition owning `rel`, or None when out of scope."""
    if rel.startswith(EXCLUDED):
        return None
    ext = os.path.splitext(rel)[1]
    if rel.startswith(TEST_PREFIXES):
        return "tests" if ext in C_EXTS or ext == ".py" else None
    if ext == ".py":
        return "python" if rel.startswith(PY_PREFIXES) else None
    if ext not in C_EXTS:
        return None
    if rel.startswith(OM_PREFIXES):
        return "om"
    return "core" if rel.startswith("src/") else None


def _record(part, rel, fn):
    return {
        "partition": part,
        "path": rel,
        "name": fn.name,
        "long_name": fn.long_name,
        "ccn": fn.cyclomatic_complexity,
        "nloc": fn.nloc,
        "params": fn.parameter_count,
        "line": fn.start_line,
    }


def _absorb(out, part, rel, functions):
    for fn in functions:
        key = (part, rel, fn.long_name)
        prev = out.get(key)
        # Same signature twice in one file (macro expansion, #if arms): keep the
        # worst, so a violation cannot hide behind its twin.
        if prev is None or fn.cyclomatic_complexity > prev["ccn"]:
            out[key] = _record(part, rel, fn)


def collect(root, strict=False, threads=None, tracked=None):
    """Map (partition, path, long_name) -> record for every function under root.

    threads > 1 makes lizard fork a process pool, which on spawn platforms
    requires the calling module to be import-safe; pass threads=1 from callers
    that are not main-guarded.

    `tracked` restricts the walk to git's view of the revision. Without it the
    head side would analyse build droppings an in-tree cmake leaves under src/
    while the base worktree is clean, inventing violations out of nothing.
    """
    root = Path(root).resolve()
    paths = [str(root / r) for r in ROOTS if (root / r).is_dir()]
    exts = lizard.get_extensions([] if strict else ["modified"])
    excludes = ["*/" + e + "*" for e in EXCLUDED]

    out = {}
    for info in lizard.analyze(
            paths,
            exclude_pattern=excludes,
            threads=threads or os.cpu_count() or 4,
            exts=exts,
    ):
        rel = os.path.relpath(info.filename, root)
        part = partition_for(rel)
        if part is None or (tracked is not None and rel not in tracked):
            continue
        _absorb(out, part, rel, info.function_list)
    return out


def collect_files(root, rels, strict=False):
    """collect(), restricted to an explicit set of repo-relative paths."""
    root = Path(root).resolve()
    analyzer = lizard.FileAnalyzer(lizard.get_extensions([] if strict else ["modified"]))
    out = {}
    for rel in sorted(rels):
        part = partition_for(rel)
        path = root / rel
        if part is None or not path.is_file():
            continue
        _absorb(out, part, rel, analyzer(str(path)).function_list)
    return out


def git(repo, *args):
    """Run git in `repo`, exiting with git's own message on failure."""
    done = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=False)
    if done.returncode:
        sys.exit(f"error: git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout.strip()


def changed_tests(repo, base_rev, head_rev=None):
    """`tests`-partition files touched between the revisions (or the worktree)."""
    spec = [base_rev, head_rev] if head_rev else [base_rev]
    names = git(repo, "diff", "--name-only", "--diff-filter=d", *spec).split("\n")
    return [n for n in names if n and partition_for(n) == "tests"]


def tracked_files(root):
    """The repo-relative paths git knows about under `root`."""
    return set(git(root, "ls-files").split("\n"))


def analyse(root, extra_files=(), threads=None, tracked=None):
    """Gated partitions across the tree, plus the named advisory `tests` files."""
    records = collect(root, threads=threads, tracked=tracked)
    records.update(collect_files(root, extra_files))
    return records


def worktree_analyse(repo, rev, extra_files=()):
    """Analyse `rev` in a throwaway worktree so the checkout is left alone.

    Only the paths the analysis reads are checked out; a full checkout of
    regression/ dominates the runtime and none of it is needed.
    """
    with tempfile.TemporaryDirectory(prefix="esbmc-ccn-") as tmp:
        tree = os.path.join(tmp, "tree")
        git(repo, "worktree", "add", "--detach", "--no-checkout", "--quiet", tree, rev)
        try:
            wanted = set(ROOTS) | {os.path.dirname(f) for f in extra_files}
            git(tree, "sparse-checkout", "set", *sorted(wanted))
            git(tree, "checkout", "--quiet")
            return analyse(tree, extra_files)
        finally:
            subprocess.run(["git", "-C", repo, "worktree", "remove", "--force", tree], check=False)
            subprocess.run(["git", "-C", repo, "worktree", "prune"], check=False)


def over_threshold(rec):
    """True when `rec` sits above its partition's gate (None = ungated)."""
    limit = THRESHOLDS.get(rec["partition"])
    return limit is not None and rec["ccn"] > limit


def _prior_ccn(base, by_name, key, rec):
    """The CCN this function had at base, tolerating a move or a re-signature.

    The exact key is (partition, path, long_name), so `git mv` and any parameter
    change re-key a function and would otherwise read as newly introduced --
    which would fail the gate on exactly the refactors it exists to encourage.
    Fall back to the *worst* same-named function in the partition: if complexity
    that high already existed under this name, the PR did not introduce it.
    """
    if key in base:
        return base[key]["ccn"]
    candidates = by_name.get((rec["partition"], rec["name"]))
    return max(candidates) if candidates else None


def compare(base, head):
    """Apply R1 and R2, returning (violations, advisory, per-partition budget)."""
    by_name = {}
    for rec in base.values():
        by_name.setdefault((rec["partition"], rec["name"]), []).append(rec["ccn"])

    violations, advisory = [], []
    for key, rec in head.items():
        before = _prior_ccn(base, by_name, key, rec)
        if before is not None and rec["ccn"] <= before:
            continue
        row = dict(rec, before=before)
        if over_threshold(rec):
            violations.append(row)
        elif rec["ccn"] >= ADVISORY_FLOOR:
            advisory.append(row)

    counts_before = Counter(r["partition"] for r in base.values() if over_threshold(r))
    counts_after = Counter(r["partition"] for r in head.values() if over_threshold(r))
    budget = {
        p: {
            "before": counts_before[p],
            "after": counts_after[p]
        }
        for p in PARTITIONS if counts_before[p] or counts_after[p]
    }
    return violations, advisory, budget


def budget_regressions(budget):
    """Partitions whose over-threshold population grew (rule R2)."""
    return {p: d for p, d in budget.items() if d["after"] > d["before"]}


def _table(rows, header, cell):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(cell(r)) + " |" for r in rows]


def render(violations, advisory, budget, strict_ccn=None):
    """Render the PR report as markdown."""
    strict_ccn = strict_ccn or {}
    thresholds = ", ".join(f"`{p}` > {t}" for p, t in sorted(THRESHOLDS.items()))
    md = [
        "## Cyclomatic complexity",
        "",
        f"Modified CCN, so a switch counts as one decision point. Gated: "
        f"{thresholds}. `tests` (unit/, regression/) is reported, never gated.",
        "",
    ]
    if not violations and not budget_regressions(budget):
        md += ["No function this PR adds or worsens exceeds its threshold.", ""]

    def row(r):
        before = "new" if r["before"] is None else str(r["before"])
        strict = strict_ccn.get((r["partition"], r["path"], r["long_name"]))
        return [
            f"`{r['name']}`",
            f"{r['path']}:{r['line']}",
            r["partition"],
            f"{before} → **{r['ccn']}**",
            str(strict) if strict is not None else "—",
        ]

    header = ["function", "location", "partition", "CCN", "strict"]
    if violations:
        md += [f"### Over threshold ({len(violations)})", ""]
        md += _table(sorted(violations, key=lambda r: -r["ccn"]), header, row) + [""]
    if advisory:
        shown = sorted(advisory, key=lambda r: -r["ccn"])[:50]
        md += [
            f"<details><summary>Below threshold but rising "
            f"({len(advisory)})</summary>",
            "",
        ]
        md += _table(shown, header, row) + ["", "</details>", ""]
    if budget:
        md += ["### Functions over threshold, repo-wide", ""]
        md += _table(
            sorted(budget.items()),
            ["partition", "before", "after", "Δ"],
            lambda kv: [
                kv[0],
                str(kv[1]["before"]),
                str(kv[1]["after"]),
                f"{kv[1]['after'] - kv[1]['before']:+d}",
            ],
        ) + [""]
    return "\n".join(md)


def render_snapshot(records):
    """Render the whole-tree distribution and worst offenders as markdown."""
    ranked = sorted(records.values(), key=lambda r: -r["ccn"])
    ccns = sorted(r["ccn"] for r in ranked)
    n = len(ccns)
    if not n:
        return "## Cyclomatic complexity snapshot\n\nNo functions found."
    md = [
        "## Cyclomatic complexity snapshot",
        "",
        f"{n} functions across the gated partitions ({', '.join(sorted(THRESHOLDS))}); "
        "modified CCN.",
        "",
    ]
    md += _table(
        [(t, sum(1 for c in ccns if c > t)) for t in (10, 15, 20, 25, 30, 50, 100)],
        ["CCN >", "functions", "share"],
        lambda kv: [str(kv[0]), str(kv[1]), f"{100 * kv[1] / n:.1f}%"],
    )
    md += [
        "",
        f"median {ccns[n // 2]}, p90 {ccns[int(0.9 * n)]}, p99 {ccns[int(0.99 * n)]}",
        "",
        "### Top 25",
        "",
    ]
    md += _table(
        ranked[:25],
        ["CCN", "function", "location", "partition"],
        lambda r: [
            str(r["ccn"]),
            f"`{r['name']}`",
            f"{r['path']}:{r['line']}",
            r["partition"],
        ],
    )
    return "\n".join(md)


def run_diff(repo, args):
    """Compare the two revisions; return (markdown, report, failed)."""
    base_rev = args.base or git(repo, "merge-base", "origin/master", "HEAD")
    tests = changed_tests(repo, base_rev, args.head)
    base = worktree_analyse(repo, base_rev, tests)
    head = (worktree_analyse(repo, args.head, tests)
            if args.head else analyse(repo, tests, tracked=tracked_files(repo)))

    # A checkout that half-materialised yields no functions, hence no
    # violations, hence a green gate reading "nothing exceeds its threshold".
    # Same for a file lizard fails to parse on one side only: it reports the
    # failure on stderr and returns an empty function list. Refuse to report on
    # a head that lost a meaningful share of the base's functions.
    if len(head) * 2 < len(base):
        sys.exit(f"error: analysed {len(head)} functions at head vs {len(base)} at "
                 f"base -- the checkout looks incomplete, refusing to report")

    violations, advisory, budget = compare(base, head)

    # Strict McCabe is context only, so pay for it on the flagged files. It
    # reads the checkout, so it is only meaningful when the checkout is head.
    strict_ccn = {}
    if not args.head:
        strict_ccn = {
            k: r["ccn"]
            for k, r in collect_files(repo, {r["path"]
                                             for r in violations}, strict=True).items()
        }
    report = {
        "mode": "diff",
        "base": base_rev,
        "head": args.head or "worktree",
        "thresholds": THRESHOLDS,
        "violations": violations,
        "advisory": advisory,
        "budget": budget,
    }
    failed = bool(violations) or bool(budget_regressions(budget))
    return render(violations, advisory, budget, strict_ccn), report, failed


def run_snapshot(repo):
    """Rank the gated partitions; return (markdown, report, failed)."""
    records = analyse(repo, tracked=tracked_files(repo))
    ranked = sorted(records.values(), key=lambda r: -r["ccn"])
    return render_snapshot(records), {"mode": "snapshot", "functions": ranked}, False


def main(argv=None):
    """Entry point; returns the process exit status."""
    ap = argparse.ArgumentParser(
        description="Cyclomatic-complexity report and ratchet for ESBMC PRs.")
    ap.add_argument("--repo", default=".", help="repository root")
    ap.add_argument("--base", help="base revision (default: merge base with master)")
    ap.add_argument("--head", help="head revision (default: the working tree)")
    ap.add_argument("--snapshot", action="store_true", help="rank the whole tree")
    ap.add_argument("--gate", action="store_true", help="exit 1 on a violation")
    ap.add_argument("--json", dest="json_out", help="write the full report here")
    ap.add_argument("--markdown", dest="md_out", help="write the summary here")
    args = ap.parse_args(argv)

    repo = os.path.abspath(args.repo)
    markdown, report, failed = (run_snapshot(repo) if args.snapshot else run_diff(repo, args))

    if args.md_out:
        Path(args.md_out).write_text(markdown, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(markdown)

    if failed and args.gate:
        print("::error::complexity gate: see the table above", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
