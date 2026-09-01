"""Find the commit that broke the nightly, without touching the test (#6735, tier 3 step 5).

Two subcommands, in the order the issue puts them:

``deflake``
    Re-run each failing test in isolation, several times, in the build that
    already exists. A test that passes even once is intermittent, and bisecting
    it would chase a ghost -- it is quarantined and reported instead.

``bisect``
    ``git bisect run`` over the commits since the last green nightly, with a
    single failing test as the predicate. Two results are escalated rather than
    reported as the culprit: a merge commit, which names a branch and not a
    change, and a range whose "good" end is already broken, which means the
    baseline is wrong and every answer bisect could give would be meaningless.

No agent and no patching: the output is a commit, and a human decides.
"""

import argparse
import json
import os
import re
import subprocess
import sys

SCHEMA = 1

DEFAULT_ATTEMPTS = 3

# git bisect announces its answer in this form.
FIRST_BAD = re.compile(r"^([0-9a-f]{7,40}) is the first bad commit", re.MULTILINE)


def run_ctest_test(build_dir, test, timeout=None):
    """Run exactly one ctest test by name. True when it passes."""
    done = subprocess.run(["ctest", "-R", f"^{re.escape(test)}$", "--output-on-failure"],
                          cwd=build_dir,
                          check=False,
                          capture_output=True,
                          text=True,
                          timeout=timeout)
    # ctest exits 0 when its filter matches nothing, and says so on stderr, not
    # stdout. Without this a deleted or misnamed test reads as a pass, gets
    # classified flaky, and is quarantined instead of bisected.
    if "No tests were found" in done.stdout + done.stderr:
        return False
    return done.returncode == 0


def deflake_one(test, attempts, run):
    """Run one test ``attempts`` times. Returns ``(stable_fail, passes)``.

    ``run`` takes a test name and returns True on pass, so the policy can be
    exercised without a build.
    """
    passes = sum(1 for _ in range(attempts) if run(test))
    return passes == 0, passes


def deflake(args):
    """Separate the reliably broken tests from the merely flaky ones."""
    tests = args.test or []
    if not tests:
        print("error: no --test given", file=sys.stderr)
        return 1

    stable, flaky = [], []
    for test in tests:
        is_stable, passes = deflake_one(
            test, args.attempts, lambda t: run_ctest_test(args.build_dir, t, args.timeout))
        (stable if is_stable else flaky).append({"test": test, "passes": passes})
        print(f"{test}: {passes}/{args.attempts} passed -> "
              f"{'stable failure' if is_stable else 'FLAKY'}")

    report = {
        "schema": SCHEMA,
        "attempts": args.attempts,
        "stable": [entry["test"] for entry in stable],
        "flaky": [entry["test"] for entry in flaky],
        "detail": stable + flaky,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, sort_keys=True)
            handle.write("\n")
    if args.quarantine and flaky:
        # Append rather than rewrite: quarantine is cumulative, and a test that
        # flaked last week is still suspect this week.
        known = set()
        if os.path.exists(args.quarantine):
            with open(args.quarantine, encoding="utf-8") as handle:
                known = {line.strip() for line in handle}
        with open(args.quarantine, "a", encoding="utf-8") as handle:
            for entry in flaky:
                if entry["test"] not in known:
                    handle.write(f"{entry['test']}\n")
    return 0


def git(repo, *args, check=True):
    """Run a git command in ``repo`` and return its stdout."""
    return subprocess.run(["git", "-C", repo, *args],
                          check=check,
                          capture_output=True,
                          text=True).stdout


def is_merge_commit(repo, sha):
    """True when the commit has more than one parent."""
    parents = git(repo, "rev-list", "--parents", "-n", "1", sha).split()
    return len(parents) > 2


def parse_first_bad(output):
    """Pull the culprit sha out of ``git bisect run`` output."""
    found = FIRST_BAD.search(output)
    return found.group(1) if found else ""


def classify(repo, sha, good_is_really_good):
    """Decide whether a bisect result can be reported as the culprit."""
    if not good_is_really_good:
        return "escalate", ("the failure reproduces on the last known-good commit, so the "
                            "baseline is wrong and the range says nothing")
    if not sha:
        return "escalate", "git bisect did not identify a first bad commit"
    if is_merge_commit(repo, sha):
        return "escalate", (f"bisect landed on merge commit {sha[:10]}, which names a branch "
                            "rather than a change")
    return "culprit", f"{sha[:10]} is the first commit where the test fails"


def predicate_at(repo, predicate, sha):
    """Run the predicate with ``sha`` checked out, then put the tree back."""
    original = git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    if original == "HEAD":  # detached
        original = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "checkout", "-q", sha)
    try:
        return subprocess.run(predicate, shell=True, cwd=repo, check=False).returncode == 0
    finally:
        git(repo, "checkout", "-q", original)


def format_comment(report):
    """Render the bisect outcome as a comment for the nightly issue."""
    if report["action"] != "culprit":
        return f"Automated bisect stopped without a culprit: {report['reason']}\n"
    return ("Automated bisect points at "
            f"`{report['commit']}`:\n\n"
            f"> {report.get('subject', '')} — {report.get('author', '')}\n\n"
            "Nothing has been changed; reverting or fixing is a human call.\n")


def bisect(args):
    """Bisect one failing test between a known-good and a known-bad commit."""
    # Confirm the ends before spending builds on the middle: a predicate that
    # fails at 'good' too makes every bisect step meaningless. This has to
    # check the commit out -- evaluating it against the current tree would just
    # re-measure 'bad'.
    good_ok = (predicate_at(args.repo, args.predicate, args.good) if args.verify_good else True)

    sha, output = "", ""
    if good_ok:
        git(args.repo, "bisect", "reset", check=False)
        git(args.repo, "bisect", "start", args.bad, args.good)
        done = subprocess.run(["git", "-C", args.repo, "bisect", "run", "sh", "-c",
                               args.predicate],
                              check=False,
                              capture_output=True,
                              text=True)
        output = done.stdout + done.stderr
        sha = parse_first_bad(output)
        git(args.repo, "bisect", "reset", check=False)

    action, reason = classify(args.repo, sha, good_ok)
    report = {
        "schema": SCHEMA,
        "action": action,
        "reason": reason,
        "commit": sha,
        "good": args.good,
        "bad": args.bad,
    }
    if sha and action == "culprit":
        report["subject"] = git(args.repo, "log", "-1", "--format=%s", sha).strip()
        report["author"] = git(args.repo, "log", "-1", "--format=%an", sha).strip()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, sort_keys=True)
            handle.write("\n")
    if args.log:
        with open(args.log, "w", encoding="utf-8") as handle:
            handle.write(output)
    if args.comment:
        with open(args.comment, "w", encoding="utf-8") as handle:
            handle.write(format_comment(report))

    print(f"{action}: {reason}")
    return 0


def main(argv=None):
    """Dispatch to deflake or bisect."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    shake = sub.add_parser("deflake", help="re-run failing tests to spot intermittent ones")
    shake.add_argument("--test", action="append", help="failing test name (repeatable)")
    shake.add_argument("--build-dir", default="build", help="configured build to run in")
    shake.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    shake.add_argument("--timeout", type=float, help="per-run timeout in seconds")
    shake.add_argument("--quarantine", help="file to append newly flaky tests to")
    shake.add_argument("--json", help="machine-readable report to write")
    shake.set_defaults(func=deflake)

    hunt = sub.add_parser("bisect", help="git bisect one test over a commit range")
    hunt.add_argument("--repo", default=".", help="repository to bisect in")
    hunt.add_argument("--good", required=True, help="last green commit")
    hunt.add_argument("--bad", required=True, help="commit the nightly failed at")
    hunt.add_argument("--predicate",
                      required=True,
                      help="shell command exiting 0 when the commit is good")
    hunt.add_argument("--verify-good",
                      action="store_true",
                      help="check the predicate passes at --good before bisecting")
    hunt.add_argument("--json", help="machine-readable report to write")
    hunt.add_argument("--log", help="raw git bisect output to write")
    hunt.add_argument("--comment", help="Markdown comment for the nightly issue")
    hunt.set_defaults(func=bisect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
