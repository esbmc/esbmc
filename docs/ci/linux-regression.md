# Linux PR regression CI

The Linux leg of the PR workflow (`.github/workflows/ci-pull-request.yml`) runs
in one of two modes. `linux-test-summary` is the check that decides the Linux
tests in both.

| Mode | Jobs | CPython check |
|---|---|---|
| `legacy` | `build-ubuntu` builds and runs `ctest -LE python-intensive` in one job | inside `build-ubuntu` |
| `parallel` | `linux-build` builds once and exports a bundle; `linux-regression` runs six shards from it | separate `cpython-check` job |

`legacy` is the default until the parallel path has been compared and
accepted (see [Activation](#activation)).

## Selecting the mode

`changes` resolves the mode once per run with
`scripts/ci/linux_tests.py resolve-mode`:

- `LINUX_TEST_DEFAULT_MODE`, in the workflow's `env:`, is the versioned
  default;
- the repository variable `LINUX_TEST_MODE`, when set to `legacy` or
  `parallel`, overrides it;
- a run that receives no variable, as a fork's may not, uses the default.

Any other value is published as the mode `invalid` instead of failing
`changes`, so the jobs gated on `changes` (doxygen, LD) still run.
`build-ubuntu` runs for every mode but `parallel`, so an invalid mode or a
failed `changes` never leaves a PR without tests; `linux-test-summary` fails
in both cases.

## What runs where

The PR baseline is `ctest -LE python-intensive` after `cmake
-DESBMC_REGRESS_TIMEOUT=120 build`, which keeps the 120s budget, the 2x/3x
exceptions in `regression/CMakeLists.txt` and the 30s CTest grace.

| Executor | CTest selection |
|---|---|
| `producer` (in `linux-build`) | `-LE '^(regression)$'`: unit tests and other tests without the `regression` label |
| `python-1`..`python-4` | `-L` the Python suite labels, `-LE '^(python-intensive/)$'`, `-I i,,4` |
| `native-1`..`native-2` | `-L '^(regression)$'`, `-LE` the Python labels and `python-intensive/`, `-I i,,2` |
| `baseline` (comparison only) | `-L '^(regression)$' -LE '^(python-intensive/)$'` |

The Python labels and shard counts live in `scripts/ci/ci_selection.py`.
`linux_tests.py plan` asks CTest for each selection and fails unless the
producer and the shards cover the baseline exactly once; it also fails on a
test with `DEPENDS`, `FIXTURES_*`, `RESOURCE_*`, `RUN_SERIAL` or `PROCESSORS`,
which striping ignores. Several `-LE` options are ANDed by CTest, which is why
each exclusion is one anchored regex.

CTest's `-I i,,k` does not select nothing when a family has fewer than `k`
tests: it repeats tests of another shard. The partition check rejects that.

## The bundle

`linux_tests.py bundle` packs, relative to the workspace:

- every `CTestTestfile.cmake` and the files they include, Catch2 discovery
  files among them;
- the executables the shard commands name inside the build tree, and the
  shared libraries of the build tree they load;
- `ci-bundle/`: metadata (checkout SHA, run, attempt, CTest version, image,
  plan digest, workspace path).

Unit-test executables stay out: their tests run on the producer. CTest records
absolute paths, so `linux_tests.py restore` refuses a bundle built in another
workspace path, a digest other than the producer's, sources checked out at
another commit than the bundle's, member paths that leave the workspace, a
tool that is not executable, and a library `ldd` cannot resolve. `pr-bundle`
needs `ubuntu-24.04`, the runner the shards use.

Before running, `linux_tests.py preflight` checks the plan digest, that the
restored tree selects exactly the planned tests with the same commands,
labels, environment and budgets, and that the harness's dynamic capabilities
(`host_cxx_headers`, `mfp8`, ...) and `z3` on `PATH` match the producer. A
mismatch fails the shard; it never becomes a new skip.

`scripts/ci/linux-runtime.sh` installs what a shard runner needs (Z3, JDK,
multilib, the Python packages); `scripts/build.sh` takes its Python packages
from the same script.

## Results and the summary

Each executor runs `ctest ... -j2 --no-tests=error --output-junit` under
`linux_tests.py run`, which samples the job's cgroup memory once a second and
writes `report.json` into the artifact `linux-result-<executor>-<attempt>`.
A run passes only when:

- CTest exited 0;
- every selected test has exactly one JUnit case, and no other case exists;
- each case passed, or was skipped with `SKIP_RETURN_CODE=10` for a reason
  the harness gives: a mode outside `--modes`, or a `REQUIRES` capability
  absent from the build's static set or the producer's probes.

A failure, a `Timeout`, a "Not Run" (missing executable), a repeated name or a
truncated file fails it. The JUnit counters are not used: CTest counts a
missing executable as skipped.

`linux-test-summary` (always runs) fails unless the jobs of the mode
succeeded and, in parallel mode, the producer published its plan and bundle
digests and the latest report of the producer and of
every shard carries the producer's plan digest and bundle digest with no
problem. Re-running a failed shard reuses the producer's bundle and publishes
a new attempt; the latest attempt counts even when it fails. An expired
bundle means a new run.

## Manual comparison

`.github/workflows/ci-linux-regression-compare.yml` (Actions >
"Linux regression comparison" > Run workflow) takes a full `candidate-sha`
and an optional `cold-build`. One build of the candidate feeds the
monolithic `baseline` and the six shards; `compare` fails on any test whose
outcome differs, on a missing report and on reports of another plan or
bundle, and publishes CTest seconds and memory per executor.

`cold-build: true` builds from an empty ccache (no restore, no save) to
measure the producer without cache. Each comparison has its own concurrency
group, so comparisons queued for one SHA do not cancel each other, and none
cancels a PR run. The CPython check of both workflows is
`.github/workflows/aux-cpython-check.yml`.

A `workflow_dispatch` workflow can only be run once it is on the default
branch, so the comparison is available after the infrastructure merges.

## Activation

Switching the default to `parallel` needs, in this order:

1. three comparisons on at least two SHAs, two warm and one cold, with equal
   names, outcomes and skips, no OOM or `oom_kill`, and the cold producer
   within its 180-minute limit;
2. an administrator requiring `linux-test-summary` while the old check is
   still required, and a control PR showing that a failing CPython check,
   a cancellation and a missing summary block the merge;
3. open PRs updated so they run the new workflow; then the old check removed;
4. the commit that sets `LINUX_TEST_DEFAULT_MODE: parallel`.

## Rollback

- Repository runs: set `LINUX_TEST_MODE=legacy` and start new runs.
- Every PR, forks included: commit `LINUX_TEST_DEFAULT_MODE: legacy`, update
  the PR branches and start new runs. Re-runs of old runs keep their old
  workflow and inputs.

`linux-test-summary` stays required in both modes.

## Measurements

`scripts/ci/shard_timings.py <ctest log>` reads per-test durations from a
CTest log and simulates the shards as `-I i,,k` stripes them. On the PR job
112391675659 (`Run tests` 4h46m45s, 23,067 tests) it simulates 51.9, 52.8,
56.5 and 53.8 minutes for the Python shards and 38.8 and 33.6 for the native
ones at `-j2`. These are simulations: queueing, runtime setup, bundle
transfer and the contention of a family run alone are not in them.

## Commands

```sh
python3 -m unittest discover -s scripts/ci/tests -p 'test_*.py'
python3 scripts/ci/linux_tests.py plan --build-dir build --out plan.json
python3 scripts/ci/shard_timings.py job.log
```
