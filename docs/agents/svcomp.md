# SV-COMP Benchmarking

The `Run Benchexec` workflow is the only measurement of ESBMC's competition
score. A number from it is easy to read as a verdict on the PR in front of you
when it is nothing of the sort.

**Label the PR.** When a change can move SV-COMP verdicts — anything under
`goto-symex/`, `solvers/`, `pointer-analysis/`, `goto-programs/`, a frontend's
semantics, an operational model, or `scripts/competitions/svcomp/` — add
`needs-svcomp-run` alongside the area label, so it is not merged on the
regression suite alone. Add `SV-COMP` when the competition setup itself is what
changed.

**Check the provenance of both runs before attributing a score move.**
`gh run list --workflow "Run Benchexec"` prints each run's `headBranch`: runs
described as "master" are routinely another PR's branch. The timeout and
`ESBMC_OPTS` are `workflow_dispatch` inputs, recorded together with the CPU and
RAM in the `<result …>` element of every `*.results.*.xml.bz2` in the
`esbmc-result` artifact — a 30s run and a 900s run are not comparable, and
neither are two different strategies. Master moves daily, so a baseline more
than a few days older than the PR run measures master's drift rather than the
PR.

**Attribute per task, not per total.** Download both `esbmc-result` artifacts,
parse the per-task `status` and `category` out of the XML, and diff the task
sets. If the tasks a PR appears to lose are the same ones another branch's run
already lost, the PR is not the cause; a third run from the same week on an
unrelated branch settles it.

**Read a per-task log before concluding the verifier regressed.**
`*.logfiles.zip` in the artifact holds ESBMC's full stdout per task.
`VERIFICATION FAILED` followed by `Unknown` is `esbmc-wrapper.py` failing to
classify the output, not ESBMC failing to find the bug.

**Changing what ESBMC prints is an interface change.** `parse_result()` in
`scripts/competitions/svcomp/esbmc-wrapper.py` classifies each task by matching
substrings of ESBMC's output, so a PR that adds, renames, or reformats a verdict
line, a property comment, or a summary block must be checked against it —
`python3 scripts/competitions/svcomp/test_esbmc_wrapper.py` covers the parsing —
and carries `needs-svcomp-run`. PR #7064 added a per-property table listing
every property including the unchecked ones; the wrapper read those rows and
turned ~2600 correct-false verdicts into `Unknown` (#7250).
