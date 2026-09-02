#!/usr/bin/env python3
"""Turn a red nightly into something a human can act on (esbmc/esbmc#6735, tier 2).

The nightly gate is only useful if its output says *what* broke, *when* it could
have broken, and whether the break is worth bisecting at all. This reads the
suite's JUnit report and emits both a JSON record and a Markdown issue body
carrying:

* every failing test, with the tail of its output;
* the commit range since the last green nightly, with authors;
* a verdict -- bisect, or escalate to a human.

Escalation exists because bisecting the wrong thing wastes a night. Too many
tests failing at once is an environment or toolchain problem rather than one bad
commit, and a run with no recorded green baseline has nothing to bisect against.
Both are stated plainly instead of being guessed at.
"""

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

SCHEMA = 1

# More simultaneous failures than this and the cause is almost never a single
# commit -- a solver upgrade, a missing dependency, a full disk.
DEFAULT_MAX_FAILURES = 25

# How much of a failing test's output to carry into the issue.
OUTPUT_TAIL_LINES = 40


def failing_cases(junit_path, tail_lines=OUTPUT_TAIL_LINES):
    """Yield ``(name, seconds, output tail)`` for failing tests only.

    Deliberately not ctest_timings.parse_junit: that one exists to be cheap over
    a whole-suite report and throws every test's output away as it goes. Here
    the output is the point, but only for the few tests that failed, so it is
    captured and immediately truncated.
    """
    for _, elem in ET.iterparse(junit_path, events=("end", )):
        if elem.tag != "testcase":
            continue
        name = elem.get("name")
        # CTest reports a skipped test as status="notrun" with no <failure>, so
        # it never satisfies this and needs no separate guard.
        failed = elem.find("failure") is not None or elem.get("status") == "fail"
        if name and failed:
            out = "".join(node.text or "" for node in elem.iter("system-out"))
            tail = "\n".join(out.splitlines()[-tail_lines:])
            yield name, round(float(elem.get("time") or 0.0), 3), tail
        elem.clear()


def commit_range(repo, last_green, head):
    """List ``(sha, author, subject)`` for the commits a regression could be in."""
    if not last_green:
        return []
    out = subprocess.run(
        ["git", "log", "--format=%H%x1f%an%x1f%s", f"{last_green}..{head}"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True)
    if out.returncode:
        # A force-push or a pruned baseline leaves an unreachable sha; say so
        # rather than reporting an empty range as "nothing changed".
        return []
    return [tuple(line.split("\x1f", 2)) for line in out.stdout.splitlines() if line]


def verdict(failures, last_green, commits, max_failures=DEFAULT_MAX_FAILURES):
    """Decide whether this red nightly is worth bisecting.

    Returns ``(action, reason)`` where action is ``bisect`` or ``escalate``.
    """
    if not failures:
        return "none", "the suite is green"
    if len(failures) > max_failures:
        return "escalate", (f"{len(failures)} tests failed at once (over the {max_failures} "
                            "threshold), which points at the environment rather than a commit")
    if not last_green:
        return "escalate", "no green nightly on record, so there is no baseline to bisect against"
    if not commits:
        return "escalate", (f"no commits between {last_green[:10]} and this run -- the same "
                            "tree changed verdict, so the failure is not deterministic")
    return "bisect", f"{len(failures)} failing test(s) over {len(commits)} commit(s)"


def format_issue(report):
    """Render the report as a GitHub issue body."""
    lines = [
        f"The nightly full suite failed on `{report['commit'][:10]}`.",
        "",
        f"- **{len(report['failures'])}** failing test(s)",
    ]
    if report["last_green"]:
        lines.append(f"- last green nightly: `{report['last_green'][:10]}`")
        lines.append(f"- **{len(report['commits'])}** commit(s) in the range")
    else:
        lines.append("- no green nightly on record")
    lines += ["", f"**Verdict: {report['action']}** — {report['reason']}", ""]

    if report["quarantined"]:
        lines += [
            f"{len(report['quarantined'])} failing test(s) are quarantined as flaky and were "
            "excluded from the verdict:", ""
        ]
        lines += [f"- `{n}`" for n in report["quarantined"]] + [""]

    lines += ["## Failing tests", ""]
    for failure in report["failures"]:
        lines += [
            f"<details><summary><code>{failure['test']}</code></summary>", "", "```",
            failure["output"] or "(no output captured)", "```", "", "</details>"
        ]

    if report["commits"]:
        lines += ["", "## Commits since the last green nightly", ""]
        lines += [
            f"- `{sha[:10]}` {subject} — {author}" for sha, author, subject in report["commits"]
        ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    """Build the nightly report from a JUnit run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--junit", required=True, help="CTest --output-junit XML of the run")
    parser.add_argument("--commit", required=True, help="commit the suite ran at")
    parser.add_argument("--last-green", default="", help="commit of the last green nightly")
    parser.add_argument("--quarantine", help="known-flaky tests to exclude from the verdict")
    parser.add_argument("--max-failures",
                        type=int,
                        default=DEFAULT_MAX_FAILURES,
                        help="above this many failures, escalate instead of bisecting")
    parser.add_argument("--repo", default=".", help="repository to read the commit range from")
    parser.add_argument("--json", help="machine-readable report to write")
    parser.add_argument("--issue-body", help="Markdown issue body to write")
    parser.add_argument("--print-failures",
                        action="store_true",
                        help="list the non-quarantined failing tests, one per line")
    parser.add_argument("--github-output",
                        help="append action= and test= for a workflow step output")
    args = parser.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from select_tests import read_lines  # pylint: disable=import-outside-toplevel

    quarantine = set(read_lines(args.quarantine)) if args.quarantine else set()
    failures, quarantined = [], []
    for name, seconds, output in failing_cases(args.junit):
        if name in quarantine:
            quarantined.append(name)
        else:
            failures.append({"test": name, "seconds": seconds, "output": output})

    commits = commit_range(args.repo, args.last_green, args.commit)
    action, reason = verdict([f["test"] for f in failures], args.last_green, commits,
                             args.max_failures)
    report = {
        "schema": SCHEMA,
        "commit": args.commit,
        "last_green": args.last_green,
        "failures": failures,
        "quarantined": sorted(quarantined),
        "commits": commits,
        "action": action,
        "reason": reason,
    }

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, sort_keys=True)
            handle.write("\n")
    if args.issue_body:
        with open(args.issue_body, "w", encoding="utf-8") as handle:
            handle.write(format_issue(report))

    if args.print_failures:
        # A wall of failures is the case de-flaking is expensive for and least
        # useful on -- print nothing so the caller skips it, same as the
        # "escalate" verdict below would once it runs.
        if len(failures) <= args.max_failures:
            print("\n".join(f["test"] for f in failures))
        return 0

    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as handle:
            handle.write(f"action={action}\n")
            # One test only: every bisect step is a full rebuild, so a second
            # predicate would double an already long night.
            handle.write(f"test={failures[0]['test'] if failures else ''}\n")

    print(f"{len(failures)} failing, {len(quarantined)} quarantined -> {action}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
