# Issues and PRs

Conventions for GitHub issues, pull requests and their labels, plus the
known-bug check to run before investigating any issue.

## PR Conventions

- Branch from `master` (the default branch)
- Target PRs to `master`
- No agent attribution in the description — see _Commit Conventions_ (AGENTS.md)
- A PR description, issue, or review comment states only what a reader can
  verify from the repository — a test, a command, a file and line. It does
  not rely on private tooling, unnamed inputs, or local runs nobody else can
  reproduce. If a claim matters enough to make, it belongs in the repository
  as a test or as steps someone can follow; if it cannot be put there, leave
  it out rather than present it as verification. (This applies even when the
  claim is true — the point is reproducibility, not honesty.)
- Fill `.github/pull_request_template.md`. Keep the description under 150
  words: what was wrong, what changed, and how it is tested. For a change
  under 20 lines, one or two sentences. Do not restate the diff, list touched
  files, or add Summary / Impact / Scope / Validation headings, bold, or
  tables.
- If you fix a bug in the same change that finds it, open only the PR. Do not
  create an issue first: the regression test is the reproducer and the PR
  description is the report.
- Before opening a PR, look for open issues it also fixes: search the error
  text and the construct, and list the area label
  (`gh issue list --repo esbmc/esbmc --label <area> --search "<terms>"`). Run
  each candidate's reproducer on your build; for every one that now passes,
  add `Fixes #N` and a `github_<N>` regression test.

## Known-Bug Check Before Working an Issue

Before investigating, reducing, or fixing any issue, check whether its
reproducer is already pinned as a KNOWNBUG test. Over 200 exist, and a new
issue is often one of them under a different description.

There is no `regression/knownbugs/` directory — `KNOWNBUG` is line 1 of a
`test.desc`, so the suite is spread across every category. Search by issue
number, by construct, and by the verdict the issue reports.

```sh
# every KNOWNBUG test in the tree
grep -rl '^KNOWNBUG' regression --include=test.desc | sed 's:/test.desc::'

# tests already tied to an issue number (find, not a glob: suites nest,
# e.g. regression/esbmc-cpp/cpp/ and regression/disabled/esbmc/)
find regression -type d -name 'github_<N>*'

# KNOWNBUG tests whose sources use the same construct
grep -rl '^KNOWNBUG' regression --include=test.desc | sed 's:/test.desc::' \
  | xargs -I{} grep -rls '<construct>' {}

# narrow by the verdict the issue wants: a false alarm is pinned by a KNOWNBUG
# expecting SUCCESSFUL, a missed bug by one expecting FAILED
grep -rl '^KNOWNBUG' regression --include=test.desc \
  | xargs grep -l 'VERIFICATION SUCCESSFUL'
```

**Run the candidate first.** A KNOWNBUG test that passes ctest is the bug
still being live; one that fails with
`passed but is marked as KNOWNBUG. Consider reclassifying it as CORE.`
(`testing_tool.py` exit 77, which ctest reports as a failure —
`SKIP_RETURN_CODE` is 10) means the bug is already gone on master, and the
issue can be closed by flipping that test to CORE. The exception is a timeout,
which KNOWNBUG accepts as satisfying the expectation — see the
`accepted under KNOWNBUG` caveat in `docs/agents/regression-tests.md`.

**The bar for "same bug" is high — be conservative.** Two reproducers are the
same bug only when one defect in ESBMC explains both: same component, same
reason for the wrong verdict. A shared surface is not evidence — the same
`VERIFICATION FAILED` line, the same crash site, the same construct, or the same
file touched are all equally consistent with two distinct defects; a
counterexample or `--goto-functions-only` dump that diverges between the two
inputs is evidence they are not one bug. When it is unclear, treat them as
distinct: a wrongly merged issue buries a live bug, while a wrongly separate
one costs one duplicate test directory.

When it is a match:

- Comment on the issue naming the existing test path (e.g.
  `regression/python/github_7552_len`), and say what makes it the same defect,
  not merely the same symptom.
- Do not add a second reproducer for it. A `github_<N>` directory that
  re-pins a defect another KNOWNBUG already covers is a duplicate.
- When you fix it, flip the existing test from KNOWNBUG to CORE in that same
  PR rather than adding a parallel directory, and add its `VERIFICATION FAILED`
  counterpart — see *Regression tests come in pairs*
  (`docs/agents/regression-tests.md`).
- Do not write the annotation into `test.desc`: line 4+ have no comment syntax,
  so every line there is a required regex. Put the issue number in the
  reproducer source (`main.c` / `main.py`) and in the commit message.

A match under `regression/disabled/` is not a gate: that tree is absent from the
`REGRESSIONS` list in `regression/CMakeLists.txt`, so nothing in it runs. Treat
it as a record that the reproducer exists, and re-home the test into a live
suite when you fix the bug.

## Issue Conventions

- Before opening an issue, search open and closed issues for the error text,
  the construct, and the flags involved
  (`gh search issues --repo esbmc/esbmc --include-prs "<terms>"`). If one
  matches, add your reproducer there as a comment instead. Search the KNOWNBUG
  tests in the same pass — see *Known-Bug Check Before Working an Issue*; a
  reproducer pinned there may have no issue open against it at all.
- Give the fields the matching form in `.github/ISSUE_TEMPLATE/` asks for: a
  reproducer (program, command, output, version) plus at most 150 words. Put
  root-cause analysis in the PR that fixes it, once.
- Questions about using ESBMC go to GitHub Discussions (Q&A), not issues.

## Issue and PR Labels

Always apply at least one label when creating an issue or PR. Pick the label
that matches the affected area — e.g. `python`, `clang-c-frontend`, `solver`,
`build`, `docs`. Use `gh label list --repo esbmc/esbmc` to see the available
labels, then `gh issue edit <N> --add-label <label>` or
`gh pr edit <N> --add-label <label>`. If no existing label fits, ask the user
rather than creating a new one.

Add `needs-svcomp-run` on top of the area label whenever the change can move
competition verdicts — see `docs/agents/svcomp.md` for what qualifies.
