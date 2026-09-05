# Fast-lane CI data

Data behind the two-tier CI of [#6735](https://github.com/esbmc/esbmc/issues/6735).
Everything here is generated and committed by a workflow; nothing is hand-edited.

Rollout steps 1, 2 and 5. The PR lane still runs the full suite; what is new
is that the nightly now *gates* master and hunts the commit behind a
regression.

| File | Written by | What it is |
| --- | --- | --- |
| `test-timings.json` | `nightly.yml` | measured duration of every test |
| `selected-tests-<year>-W<week>.txt` | `nightly.yml` (first run of the week) | the week's fast-lane sample |
| `core-set.txt` | `core-set.yml` (monthly) | always-run tests, solved from coverage |
| `last-green-nightly.txt` | `nightly.yml` | commit the bisect range starts from |
| `quarantine.txt` | `nightly.yml` | tests found intermittent, excluded from the verdict |

## Reproducing a fast-lane failure locally

The selection is a pure function of the ISO week, so it is the same for every PR
that week and can be regenerated exactly:

```sh
scripts/ci/select_tests.py --week 2026-W36 --output /tmp/selected.txt
scripts/ci/run_selected_tests.py --build-dir build --tests /tmp/selected.txt --output-on-failure
```

`select_tests.py` defaults to `--budget-seconds 900 --jobs 2`, matching the PR
Linux leg. Pass `--always-run ci/core-set.txt` to include the core set as CI
does. Any argument `run_selected_tests.py` does not recognise is forwarded to
`ctest`.

## How the sample is drawn

Tests are added until the projected wall-clock hits the budget — measured
cumulative runtime, not a test count. Sampling is stratified twice: by suite
(the ctest label), so no frontend or solver backend goes dark for a week, and
within a suite by runtime tercile, so slow tests are not squeezed out by cheap
ones. Every suite contributes at least one test even when its proportional share
rounds to nothing.

A test with no measurement yet — newly added, or skipped on the measuring host —
is costed at its suite's median rather than zero, so an unmeasured suite cannot
be sampled without bound.

## Caveats

- **Durations are relative, not absolute.** They are measured on a self-hosted
  runner at full parallelism, while the PR lane runs `-j2` on a hosted runner.
  `--packing-efficiency` absorbs the systematic part; shadow mode (rollout step
  3) is where the projection gets checked against reality.
- **Deleted tests linger** in `test-timings.json`, because the nightly merges
  rather than replaces. `run_selected_tests.py` reports and ignores selected
  tests that the build does not contain.
- **`core-set.txt` is built from a `CORE_REGRESSION_ONLY` coverage build**, so
  `KNOWNBUG` and `FUTURE` tests are never candidates for the always-run set.

## The nightly gate (tier 2 and rollout step 5)

`nightly.yml` runs the full suite once a night against master. One run serves
both tiers: it gates, and its JUnit report is also what the timing table is
built from. Measuring and gating separately would mean paying for the four-hour
suite twice a night.

When it goes red, the run does three things before anyone wakes up.

**Confirms.** Each failing test is re-run in isolation, in the build that still
exists, three times by default. A test that passes even once is intermittent:
it is appended to `quarantine.txt`, excluded from the verdict, and never
bisected — bisecting a flaky test finds a commit at random. De-flaking happens
in the suite job precisely because the build is still there; rebuilding to
re-run three tests would cost half an hour.

**Decides whether bisecting is worth it.** Three situations are escalated to a
human instead:

| Situation | Why not bisect |
| --- | --- |
| more than 25 tests failing at once | points at the environment or toolchain, not one commit |
| no green nightly on record | there is no baseline to bisect against |
| no commits since the last green | the same tree changed verdict, so the failure is not deterministic |

**Bisects.** `git bisect run` over the commits since the last green nightly,
with the failing test as the predicate. Two outcomes are escalated rather than
reported as the culprit: a merge commit, which names a branch rather than a
change, and a range whose "good" end is already broken, which would make every
answer meaningless. A commit that fails to build exits 125 and is skipped, not
blamed.

The result is an issue naming the commit, its subject, and its author (resolved
to a GitHub handle so the ping actually reaches them). **Nothing is patched and
nothing is merged** — reverting or fixing is a human call, which is where
rollout step 5 deliberately stops.

### What it costs

#6735 estimates the bisect at `O(log n) × single-test-time`, "typically
minutes". That counts only the test. Each bisect step is a full ESBMC rebuild,
so ~20 commits is around five rebuilds — hours, not minutes, though ccache
absorbs most of it and the job has all night. Only one failing test is bisected
per run for the same reason; the rest are listed in the issue.

### Reproducing locally

```sh
scripts/ci/nightly_report.py --junit junit.xml --commit HEAD \
    --last-green "$(grep -v '^#' ci/last-green-nightly.txt)" --issue-body /tmp/issue.md

BISECT_TEST='regression/esbmc/some_test' scripts/ci/nightly_bisect.py bisect \
    --good <sha> --bad HEAD --predicate scripts/ci/bisect_predicate.sh --verify-good
```
