# Fast-lane CI data

Data behind the two-tier CI of [#6735](https://github.com/esbmc/esbmc/issues/6735).
Everything here is generated and committed by a workflow; nothing is hand-edited.

Rollout steps 1–3. **Nothing gates on them yet**: the PR lane still runs the
full suite, and shadow mode only measures what a fast lane *would* have missed.

| File | Written by | What it is |
| --- | --- | --- |
| `test-timings.json` | `test-timings.yml` (nightly) | measured duration of every test |
| `selected-tests-<year>-W<week>.txt` | `test-timings.yml` (first run of the week) | the week's fast-lane sample |
| `core-set.txt` | `core-set.yml` (monthly) | always-run tests, solved from coverage |

Shadow mode holds no committed data — its per-PR reports are run artifacts, and
`shadow-aggregate.yml` posts the running total to #6735 weekly.

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

## Shadow mode (rollout step 3)

The Linux PR leg runs the week's subset first, cold and timed, then the full
suite, then compares the two. It answers the two questions step 4 needs:

- **What does the fast lane really cost?** A projection from durations measured
  on another machine at another parallelism is not an answer; running it is.
- **What would it have missed?** A test that failed in the full run and was
  never sampled is an escaped regression.

A test the fast lane *ran and passed* that then failed in the full run is
counted as **unstable**, not escaped — it is flaky or order-dependent, and
blaming the sampling for it would overstate the escape rate. The issue calls
flakiness out as the thing most likely to derail the later bisect loop, so it is
worth counting on its own from the start.

The shadow selection excludes `regression/python-intensive/`, because the PR leg
does too and the comparison is only meaningful over one universe. Same week
seed, so it is the nightly's draw minus that suite.

Both the shadow steps and the digest workflow are temporary. Remove
`shadow-fast-lane` from `pull_request.yml` and delete `shadow-aggregate.yml`
once the rate is known and step 4 is decided.
