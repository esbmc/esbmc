# Linux PR regression CI

The Linux tests of the PR workflow (`.github/workflows/ci-pull-request.yml`)
run as:

```
linux-build (build, tests without the regression label, bundle) ─┐
cpython-check (scripts/check_python_tests.sh) ────────────────────┴─► 7 module jobs ─► linux-test-summary
```

- `linux-build` and `cpython-check` start together.
- The module jobs (`linux-regression`) start only when both passed, and run
  the regression from the bundle `linux-build` exported, without rebuilding.
- `linux-test-summary` always runs and is the required check of the Linux
  tests. It fails unless the build, the CPython check and every module job
  succeeded, and every report holds the producer's plan and bundle with no
  problem.

This replaces the `static llvm-22 DebugOpt / Build ESBMC (ubuntu-24.04)` job,
which built and ran the whole suite in one job.

## What runs where

The PR baseline is `ctest -LE python-intensive` after `cmake
-DESBMC_REGRESS_TIMEOUT=120 build`, which keeps the 120s budget, the 2x/3x
exceptions in `regression/CMakeLists.txt` and the 30s CTest grace.

| Executor | CTest selection |
|---|---|
| `producer` (in `linux-build`) | `-LE '^(regression)$'`: unit tests and other tests without the `regression` label |
| `numpy-1`, `numpy-2` | `-L '^(numpy/)$'`, `-I i,,2` |
| `python-1`, `python-2` | `-L '^(python/)$'`, `-I i,,2` |
| `python-benchmarks` | quixbugs, humaneval, mopsa, consensus, ai-generated-code, python-coverage, python-contracts |
| `cpp` | `-L '^esbmc-cpp[0-9]*/.*$'`: every `esbmc-cpp*` suite |
| `c-and-tools` | every other `regression` test, so a new suite is never left out |

Every module also excludes `python-intensive/`. The modules and their shard
counts live in `MODULES` in `scripts/ci/ci_selection.py`; NumPy and Python
are split in two because they hold most of the test time.
`linux_tests.py plan` asks CTest for each selection and fails unless the
producer and the shards cover the baseline exactly once; it also fails on a
test with `DEPENDS`, `FIXTURES_*`, `RESOURCE_*`, `RUN_SERIAL` or `PROCESSORS`,
which striping ignores. Several `-LE` options are ANDed by CTest, which is why
each exclusion is one anchored regex.

CTest's `-I i,,k` does not select nothing when a module has fewer than `k`
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

`linux-test-summary` fails unless the build, the CPython check and every
module job succeeded, the producer published its plan and bundle digests,
and the latest report of the producer and of
every shard carries the producer's plan digest and bundle digest with no
problem. Re-running a failed shard reuses the producer's bundle and publishes
a new attempt; the latest attempt counts even when it fails. An expired
bundle means a new run.

## Required check and rollback

Branch protection must require `linux-test-summary` instead of
`static llvm-22 DebugOpt / Build ESBMC (ubuntu-24.04)`, which no longer runs.
Rolling back means reverting the commits that replaced that job.

## Sizing

The modules were sized from the per-test durations of PR job 112391675659
(`Run tests` 4h46m45s, 23,067 tests), striped as `-I i,,k` does and run on two
slots:

| Module job | Simulated minutes |
|---|---|
| `numpy-1`, `numpy-2` | 53.8, 53.8 |
| `python-1`, `python-2` | 38.7, 38.3 |
| `c-and-tools` | 48.0 |
| `python-benchmarks` | 26.6 |
| `cpp` | 24.3 |

These are simulations: queueing, runtime setup, bundle transfer and the
contention of a module run alone are not in them.

## Commands

```sh
python3 -m unittest discover -s scripts/ci/tests -p 'test_*.py'
python3 scripts/ci/linux_tests.py plan --build-dir build --out plan.json
```
