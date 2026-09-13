# ESBMC known-bug (KNOWNBUG) triage report — 2026-09-10

Every test in the regression suite marked `KNOWNBUG` on line 1 of its `test.desc`,
classified by what the defect actually *is* and by whether anything tracks it.
Each test appears in exactly one category and exactly one tracking state.

## 1. Scope

- **220 KNOWNBUG tests** across **32 suites**.
- There is no `regression/knownbugs/` directory. `KNOWNBUG` is a `test.desc` mode, so the
  set is spread across every suite and has to be enumerated rather than listed.
- **11** of them sit under `regression/disabled/`, which is absent from the
  `REGRESSIONS` list in `regression/CMakeLists.txt` and therefore never runs.
- `FUTURE` tests (missing implementation, not defects) are out of scope.

## 2. Method

Each test was executed with its own `test.desc` command line — the same argument
construction `regression/testing_tool.py` performs — in a private empty working
directory, capturing stdout and stderr together, with a 30 s budget. Category
assignment is made from that captured output, not from the test's name or location.
The 28 tests that exceeded 30 s were re-run with a 300 s budget to separate *slow*
from *no answer*: 12 answered on the second pass and are classified by that answer,
leaving 16 with no result in five minutes.

Categories are applied in a fixed priority order so the partition is well defined:
`C0` (already passing) → `C1` (not measurable) → `C2`/`C3` (crash, internal error) →
`C7` (no answer) → `C4` (refusal) → `C6` (unknown) → `C5a`/`C5b` (wrong verdict) → `C8`.

### Binary provenance — read before acting on any single row

The measurements come from an `esbmc` binary taken from the shared `build/` tree at
16:55 on 2026-09-10. That build directory was in use by a concurrent session, and the
binary it produced was **not** built from `master`: the checked-out branch was
`feat/irep2-typecast-reference-arms` (master `698c6fe353` plus three unmerged `[irep2]`
commits), and it is *missing* master commits up to `2d5012082f`. Those three commits
touch `src/util/irep/migrate.cpp`, `src/util/lang/c_typecast.cpp` and `src/irep2/*` —
shared infrastructure every frontend goes through, not C++-only code.

The consequence is bounded but real: the **category distribution** is robust, because
it is dominated by frontend refusals, crashes, and the expected-verdict direction
recorded in each `test.desc`. Individual rows whose verdict could plausibly hinge on
that delta — in particular the `C0` rows and the three `migrate expr failed` rows in
`C3` — must be re-measured against a `master` build before anyone acts on them.
The repository's `Master` workflow only redeploys the website; it runs no tests, so a
green master run is not evidence either way.

### How issue linkage was determined

Every issue and pull request in `esbmc/esbmc` (7,598 records: 2,277 issues, 5,321 PRs)
was fetched and matched against each test on three tiers of evidence:

| Tier | Rule |
|---|---|
| **declared** | The directory is named `github_<N>`, or a file in the test references `#N`, `esbmc/esbmc#N`, or an issue URL. |
| **path-mention** | An issue or PR body contains the test's full path, e.g. `regression/python/foo`. |
| **name-mention (weak)** | Only the bare directory name appears, and only when that name is distinctive (contains `_`, ≥ 12 characters). Reported but never counted as tracking. |

A link to a *pull request* is provenance, not tracking: it usually just identifies the
PR that added the test. Only issues count toward tracked status.

## 3. Categories

| Category | Tests | Share |
|---|---:|---:|
| **C0** Already passing (stale KNOWNBUG) | 5 | 2% |
| **C1** Not measurable in this build | 5 | 2% |
| **C2** Crash (SIGSEGV/SIGABRT) | 2 | 1% |
| **C3** Internal error / uncaught exception | 11 | 5% |
| **C4** Unsupported construct (frontend refusal) | 33 | 15% |
| **C5a** Unsound: missed bug (false negative) | 43 | 20% |
| **C5b** Imprecise: spurious alarm (false positive) | 55 | 25% |
| **C6** Inconclusive (`VERIFICATION UNKNOWN`) | 44 | 20% |
| **C7** No answer within budget (performance) | 16 | 7% |
| **C8** Other output mismatch | 6 | 3% |
| **Total** | **220** | 100% |

**C0 — Already passing (stale KNOWNBUG).** The test's full expected output is produced today: the defect it pins appears fixed. `testing_tool.py` turns this into a *failure* (exit 77, `Consider reclassifying it as CORE`), so a registered test in this class makes the suite red.

**C1 — Not measurable in this build.** The run is stopped by build configuration or by a stale flag in `test.desc`, not by an ESBMC defect. These tests can never match their expected output, so as KNOWNBUG they pass unconditionally and would keep passing even if the underlying bug were fixed.

**C2 — Crash (SIGSEGV/SIGABRT).** ESBMC dies on a signal and prints its internal-error banner. No verdict is produced.

**C3 — Internal error / uncaught exception.** ESBMC exits through an uncaught C++ exception or an internal invariant message (`migrate expr failed`, `Non-pointer op being interpreted as pointer`, a solver-side exception). No verdict is produced.

**C4 — Unsupported construct (frontend refusal).** ESBMC stops with a diagnostic that names a construct it does not model — `PARSING ERROR`, `CONVERSION ERROR`, or an explicit *not supported* message. The program is never verified.

**C5a — Unsound: missed bug (false negative).** The test expects `VERIFICATION FAILED` and ESBMC reports `VERIFICATION SUCCESSFUL`. This is the soundness-critical class: ESBMC currently proves a program correct that the test asserts is buggy.

**C5b — Imprecise: spurious alarm (false positive).** The test expects `VERIFICATION SUCCESSFUL` and ESBMC reports `VERIFICATION FAILED` — a false alarm on a program the test asserts is correct.

**C6 — Inconclusive (`VERIFICATION UNKNOWN`).** ESBMC terminates without a verdict, almost always k-induction or bounded unwinding failing to converge inside the configured bound.

**C7 — No answer within budget (performance).** No result within **300 s**. The suite's own per-test budget is 1200 s, so a few of these may still terminate; none does so inside five minutes.

**C8 — Other output mismatch.** The verdict line matches but another required line in `test.desc` does not (a coverage percentage, a counterexample detail, a diagnostic string).

## 4. Tracking status

| Status | Tests | Share |
|---|---:|---:|
| **Tracked** | 30 | 14% |
| **Stale tracking** | 46 | 21% |
| **PR provenance only** | 48 | 22% |
| **Weak match only** | 7 | 3% |
| **Untracked** | 89 | 40% |
| **Total** | **220** | 100% |

**Tracked.** An **open** issue is linked to the test, either declared in the tree (`github_<N>` directory or an explicit `#N` / issue URL in the test's own files) or because an issue body names the test path.

**Stale tracking.** The only linked issues are **closed**, yet the test is still marked KNOWNBUG. Either the issue was closed without the bug being fixed, or the test outlived it.

**PR provenance only.** The only reference is a pull request — typically the PR that added the test. A PR is not a tracker: nothing here is on an issue list.

**Weak match only.** The test's bare directory name appears in some issue text, but not its path. Reported separately because a bare-name match is not reliable evidence.

**Untracked.** No issue and no PR anywhere in the repository's history mentions this test by path or by name.

**190 of 220 known bugs (86%) have no open issue behind them.**
Only 30 do. The largest single block is the 89 tests that no issue and no
pull request mentions at all: the `test.desc` file is the only record that the defect exists.

## 5. Category against tracking status

| Category | Tracked | Stale tracking | PR provenance only | Weak match only | Untracked | Total |
|---|---:|---:|---:|---:|---:|---:|
| **C0** | 0 | 0 | 2 | 0 | 3 | 5 |
| **C1** | 0 | 0 | 0 | 0 | 5 | 5 |
| **C2** | 0 | 1 | 0 | 0 | 1 | 2 |
| **C3** | 2 | 2 | 2 | 1 | 4 | 11 |
| **C4** | 3 | 9 | 4 | 1 | 16 | 33 |
| **C5a** | 7 | 14 | 2 | 0 | 20 | 43 |
| **C5b** | 9 | 13 | 10 | 2 | 21 | 55 |
| **C6** | 0 | 3 | 25 | 2 | 14 | 44 |
| **C7** | 9 | 3 | 2 | 1 | 1 | 16 |
| **C8** | 0 | 1 | 1 | 0 | 4 | 6 |
| **Total** | 30 | 46 | 48 | 7 | 89 | 220 |

## 6. What needs action

### 6.1 Stale KNOWNBUG — the defect appears fixed (`C0`)

These produce their full expected output. Two are registered with ctest, so the suite
is red on them right now; the other three sit in trees that never run (`disabled/`,
`windows/`), which is why nothing has flagged them.

| Test | Registered? | Tracking | Reference |
|---|---|---|---|
| `regression/disabled/ch21_3` | no | Untracked | — |
| `regression/esbmc-solidity/delegate_shadow_3` | yes — **suite is red** | PR provenance only | [#5318](https://github.com/esbmc/esbmc/pull/5318) (PR, closed) |
| `regression/esbmc-solidity/nested_array_deep_1` | yes — **suite is red** | PR provenance only | [#5318](https://github.com/esbmc/esbmc/pull/5318) (PR, closed) |
| `regression/windows/cpp_priority_queue_size_bug` | no | Untracked | — |
| `regression/windows/k-induction_cpp_stack_empty_bug` | no | Untracked | — |

Confirmed independently through ctest rather than the scan harness:

```
$ ctest -R 'regression/esbmc-solidity/(delegate_shadow_3|nested_array_deep_1)$'
ERROR: Test 'esbmc-solidity/delegate_shadow_3' passed but is marked as KNOWNBUG.
        Consider reclassifying it as CORE.
ERROR: Test 'esbmc-solidity/nested_array_deep_1' passed but is marked as KNOWNBUG.
        Consider reclassifying it as CORE.
0% tests passed, 2 tests failed out of 2
```

Both are subject to the provenance caveat in §2 — re-run them against a `master`
build before reclassifying. If they hold, they become `CORE` and gain a
`VERIFICATION FAILED` counterpart.

### 6.2 Tests that cannot fail, whatever ESBMC does (`C1`)

Each of these stops before verification for a reason that has nothing to do with the
bug it is supposed to pin. Because a KNOWNBUG test passes precisely when it does *not*
match its expected output, all five pass unconditionally — and would keep passing if
the underlying defect were fixed, so the fix would never be noticed.

| Test | Why it stops | Action |
|---|---|---|
| `regression/csmith/csmith02` | `ERROR: Invalid command line: unrecognised option '--ssa-full-names'` | Flag no longer exists anywhere in `src/`. Repair or delete the test. |
| `regression/csmith/csmith03` | same removed flag | as above |
| `regression/csmith/csmith06` | `The mathsat solver has not been built into this version` | Needs a `REQUIRES` capability line, not a silent pass. |
| `regression/cheri-128/01_cheri_ptr5_failed-hybrid` | `This build of ESBMC does not have CHERI support` | Correctly skipped by CMake here; unregistered, so harmless. |
| `regression/cheri-c/01_cheri_ptr5_failed` | same | as above |

`--ssa-full-names` returns nothing from `grep -rn 'ssa-full-names' src/` and does not
appear in `esbmc --help`, so `csmith02` and `csmith03` have been inert since it was
removed. They are registered with ctest and report `Passed` on every run.

### 6.3 Unsound results nobody is tracking (22 of 43 in `C5a`)

`C5a` is the class where ESBMC returns `VERIFICATION SUCCESSFUL` on a program the test
asserts is buggy. These are the rows in it with no issue behind them at all.

| Test | Tracking | Reference |
|---|---|---|
| `regression/cbmc/01_cbmc_BV_Arithmetic4` | Untracked | — |
| `regression/cstd/scanf_float_bug` | Untracked | — |
| `regression/cstd/scanf_float_bug_2` | Untracked | — |
| `regression/cstd/strstr_bug` | Untracked | — |
| `regression/cuda/benchmarks/099_test8` | Untracked | — |
| `regression/disabled/ch7_13` | Untracked | — |
| `regression/esbmc-cpp/cpp/dangling_temporary_ref` | PR provenance only | [#6436](https://github.com/esbmc/esbmc/pull/6436) (PR, closed) |
| `regression/esbmc-solidity/library_11` | Untracked | — |
| `regression/esbmc-solidity/send_ether_via_creation_2` | Untracked | — |
| `regression/esbmc-unix/03_wait_notify` | Untracked | — |
| `regression/esbmc-unix/03_wait_notify2` | Untracked | — |
| `regression/esbmc-unix2/11_cook.fig2.pldi07.extended` | Untracked | — |
| `regression/esbmc-unix2/11_spin2003-cex` | Untracked | — |
| `regression/esbmc-unix2/23_picosat-846_03` | Untracked | — |
| `regression/esbmc/force_malloc_success_unrepresentable` | Untracked | — |
| `regression/ir-ra/ra-neg-zero-if-merge-lost-sign` | Untracked | — |
| `regression/ir-ra/ra-neg-zero-ternary-order-lost-sign` | Untracked | — |
| `regression/llvm/compound_literal` | Untracked | — |
| `regression/loop-invariants/6-invariant_in_wrong_place` | Untracked | — |
| `regression/loop-invariants/7-not_conjunct_invariant` | Untracked | — |
| `regression/nonz3/21_printtokens2` | Untracked | — |
| `regression/python/bytes_method_not_str_fail` | PR provenance only | [#5827](https://github.com/esbmc/esbmc/pull/5827) (PR, closed) |

### 6.4 Bugs whose issue is closed while the test still pins them (`Stale tracking`)

46 tests link only to closed issues. Each is either a bug that outlived the issue
that reported it, or a test that should have been retired with it. The largest groups
are umbrella issues that were closed with the individual tests left behind:

| Closed issue | Tests still marked KNOWNBUG |
|---|---:|
| [#4397](https://github.com/esbmc/esbmc/issues/4397) [esbmc-cpp] C++ language-feature regressions marked KNOWNBUG | 5 |
| [#1092](https://github.com/esbmc/esbmc/issues/1092) K-Induction is breaking assertions inside loops. | 4 |
| [#5931](https://github.com/esbmc/esbmc/issues/5931) [Python] models/heapq.py exists but heapq is missing from the impo | 3 |
| [#7025](https://github.com/esbmc/esbmc/issues/7025) [C++] Member access via this is misresolved for a non-first base w | 2 |
| [#6961](https://github.com/esbmc/esbmc/issues/6961) [contracts] --replace-call-with-contract aborts on an assigns clau | 2 |
| [#7055](https://github.com/esbmc/esbmc/issues/7055) [contracts] a multi-level assigns target generates no VCCs, so a f | 2 |
| [#6640](https://github.com/esbmc/esbmc/issues/6640) [python] Function values are static aliases only: a runtime-chosen | 2 |
| [#4791](https://github.com/esbmc/esbmc/issues/4791) [python] regression/quixbugs/minimum_spanning_tree_fail: KNOWNBUG | 2 |

## 7. Distribution by suite

| Suite | Tests | Dominant category |
|---|---:|---|
| `k-induction` | 24 | C6 (20) |
| `esbmc-cpp` | 23 | C5b (8) |
| `python` | 21 | C5b (10) |
| `quixbugs` | 19 | C7 (10) |
| `humaneval` | 15 | C5b (8) |
| `k-induction-parallel` | 14 | C6 (13) |
| `disabled` | 11 | C6 (6) |
| `cstd` | 10 | C5b (4) |
| `function_contract` | 10 | C5a (6) |
| `numpy` | 10 | C5b (4) |
| `esbmc` | 9 | C4 (3) |
| `esbmc-solidity` | 8 | C5b (4) |
| `loop-invariants` | 8 | C5b (3) |
| `csmith` | 6 | C1 (3) |
| `esbmc-unix2` | 5 | C5a (4) |
| `esbmc-unix` | 4 | C5a (2) |
| `cvc` | 3 | C6 (2) |
| `cbmc` | 2 | C5a (2) |
| `ir-ra` | 2 | C5a (2) |
| `llvm` | 2 | C5a (1) |
| `mopsa` | 2 | C7 (1) |
| `windows` | 2 | C0 (2) |
| `bitwuzla` | 1 | C5b (1) |
| `cuda` | 1 | C5a (1) |
| `esbmc-cpp11` | 1 | C5b (1) |
| `floats-regression` | 1 | C7 (1) |
| `goto-coverage` | 1 | C8 (1) |
| `nonz3` | 1 | C5a (1) |
| `parallel-solving` | 1 | C4 (1) |
| `z3` | 1 | C4 (1) |
| `cheri-128` | 1 | C1 (1) |
| `cheri-c` | 1 | C1 (1) |

## 8. Full classification

Every KNOWNBUG test, once. Sorted by category, then path.

### C0 — Already passing (stale KNOWNBUG) (5)

| Test | Observed | Tracking | Issue / PR |
|---|---|---|---|
| `regression/disabled/ch21_3` | reports SUCCESSFUL | Untracked | — |
| `regression/esbmc-solidity/delegate_shadow_3` | reports SUCCESSFUL | PR provenance only | [#5318](https://github.com/esbmc/esbmc/pull/5318) (PR, closed) |
| `regression/esbmc-solidity/nested_array_deep_1` | reports SUCCESSFUL | PR provenance only | [#5318](https://github.com/esbmc/esbmc/pull/5318) (PR, closed) |
| `regression/windows/cpp_priority_queue_size_bug` | reports FAILED | Untracked | — |
| `regression/windows/k-induction_cpp_stack_empty_bug` | reports FAILED | Untracked | — |

### C1 — Not measurable in this build (5)

| Test | Observed | Tracking | Issue / PR |
|---|---|---|---|
| `regression/cheri-128/01_cheri_ptr5_failed-hybrid` | error exit | Untracked | — |
| `regression/cheri-c/01_cheri_ptr5_failed` | error exit | Untracked | — |
| `regression/csmith/csmith02` | error exit | Untracked | — |
| `regression/csmith/csmith03` | error exit | Untracked | — |
| `regression/csmith/csmith06` | error exit | Untracked | — |

### C2 — Crash (SIGSEGV/SIGABRT) (2)

| Test | Observed | Tracking | Issue / PR |
|---|---|---|---|
| `regression/csmith/csmith07` | crash | Untracked | — |
| `regression/python/callable_module_list` | crash | Stale tracking | [#6640](https://github.com/esbmc/esbmc/issues/6640) (issue, closed) |

### C3 — Internal error / uncaught exception (11)

| Test | Observed | Tracking | Issue / PR |
|---|---|---|---|
| `regression/esbmc-cpp/try_catch/lower-exceptions_noexcept_custom_terminate` | reports FAILED | Untracked | — |
| `regression/function_contract/github_4219_old_in_forall_array_param_replace_mode_ensures_knownbug` | error exit | Tracked | [#7057](https://github.com/esbmc/esbmc/issues/7057) (issue, open)<br>[#7627](https://github.com/esbmc/esbmc/issues/7627) (issue, open)<br>+1 more |
| `regression/function_contract/github_4219_old_in_forall_ptr_to_array_knownbug` | error exit | Tracked | [#7057](https://github.com/esbmc/esbmc/issues/7057) (issue, open)<br>[#4219](https://github.com/esbmc/esbmc/issues/4219) (issue, closed)<br>+2 more |
| `regression/humaneval/humaneval_39` | reports FAILED | Weak match only | [#4807](https://github.com/esbmc/esbmc/issues/4807) (issue, closed) |
| `regression/numpy/array_return_side_effect_edge` | reports FAILED | PR provenance only | [#7386](https://github.com/esbmc/esbmc/pull/7386) (PR, closed) |
| `regression/numpy/isclose` | error exit | Untracked | — |
| `regression/numpy/nextafter` | error exit | Untracked | — |
| `regression/numpy/remainder` | error exit | Untracked | — |
| `regression/python/bare_raise_nested` | reports FAILED | PR provenance only | [#5608](https://github.com/esbmc/esbmc/pull/5608) (PR, closed)<br>[#6468](https://github.com/esbmc/esbmc/pull/6468) (PR, closed) |
| `regression/python/callable_class_field` | uncaught exception | Stale tracking | [#6640](https://github.com/esbmc/esbmc/issues/6640) (issue, closed) |
| `regression/python/github_3765` | uncaught exception | Stale tracking | [#3765](https://github.com/esbmc/esbmc/issues/3765) (issue, closed)<br>[#7313](https://github.com/esbmc/esbmc/pull/7313) (PR, closed) |

### C4 — Unsupported construct (frontend refusal) (33)

| Test | Observed | Tracking | Issue / PR |
|---|---|---|---|
| `regression/csmith/csmith04` | refused | Untracked | — |
| `regression/cvc/quantifiers-field` | error exit | Untracked | — |
| `regression/disabled/esbmc/22_schedule2` | refused | Untracked | — |
| `regression/esbmc-cpp/cpp/bicycle` | refused | Stale tracking | [#4397](https://github.com/esbmc/esbmc/issues/4397) (issue, closed) |
| `regression/esbmc-cpp/try_catch/lower-exceptions_concurrent_dualuse` | error exit | PR provenance only | [#5244](https://github.com/esbmc/esbmc/pull/5244) (PR, closed) |
| `regression/esbmc-cpp/try_catch/lower-exceptions_thread_computed_routine` | error exit | PR provenance only | [#5261](https://github.com/esbmc/esbmc/pull/5261) (PR, closed) |
| `regression/esbmc-cpp/try_catch/lower-exceptions_thread_computed_routine_fail` | error exit | Untracked | — |
| `regression/esbmc-cpp/unix/10_bicycle_03` | refused | Stale tracking | [#4397](https://github.com/esbmc/esbmc/issues/4397) (issue, closed) |
| `regression/esbmc-unix/05_pfscan-1.0_01` | refused | Stale tracking | [#1028](https://github.com/esbmc/esbmc/issues/1028) (issue, closed) |
| `regression/esbmc-unix2/11_scull` | refused | Untracked | — |
| `regression/esbmc/fam_false_2` | refused | Untracked | — |
| `regression/esbmc/fam_true_4` | refused | Untracked | — |
| `regression/esbmc/github_197` | refused | Stale tracking | [#197](https://github.com/esbmc/esbmc/issues/197) (issue, closed) |
| `regression/humaneval/humaneval_123` | error exit | Weak match only | [#4807](https://github.com/esbmc/esbmc/issues/4807) (issue, closed) |
| `regression/humaneval/humaneval_145` | error exit | PR provenance only | [#4812](https://github.com/esbmc/esbmc/pull/4812) (PR, closed)<br>[#5390](https://github.com/esbmc/esbmc/pull/5390) (PR, closed) |
| `regression/humaneval/humaneval_148` | error exit | Untracked | — |
| `regression/humaneval/humaneval_158` | error exit | Untracked | — |
| `regression/humaneval/humaneval_87` | error exit | Stale tracking | [#7328](https://github.com/esbmc/esbmc/issues/7328) (issue, closed) |
| `regression/k-induction/github_1092_4_false` | refused | Stale tracking | [#1092](https://github.com/esbmc/esbmc/issues/1092) (issue, closed) |
| `regression/k-induction/z_sum_array` | refused | Untracked | — |
| `regression/loop-invariants/cpp_priority_queue_size_bug` | refused | Untracked | — |
| `regression/loop-invariants/cpp_stack_top_bug` | refused | Untracked | — |
| `regression/numpy/array_return_call_arg_edge` | error exit | PR provenance only | [#7386](https://github.com/esbmc/esbmc/pull/7386) (PR, closed) |
| `regression/numpy/det2` | error exit | Untracked | — |
| `regression/parallel-solving/05_pfscan-1.0_01` | refused | Untracked | — |
| `regression/python/github_6743_ambiguous` | error exit | Stale tracking | [#6743](https://github.com/esbmc/esbmc/issues/6743) (issue, closed) |
| `regression/python/sorted_key_tuple_elems_no_subscript` | error exit | Untracked | — |
| `regression/quixbugs/minimum_spanning_tree` | error exit | Stale tracking | [#4778](https://github.com/esbmc/esbmc/issues/4778) (issue, open)<br>[#4790](https://github.com/esbmc/esbmc/issues/4790) (issue, closed)<br>+6 more |
| `regression/quixbugs/minimum_spanning_tree_fail` | error exit | Stale tracking | [#4778](https://github.com/esbmc/esbmc/issues/4778) (issue, open)<br>[#4791](https://github.com/esbmc/esbmc/issues/4791) (issue, closed) |
| `regression/quixbugs/rpn_eval_fail` | error exit | Tracked | [#4778](https://github.com/esbmc/esbmc/issues/4778) (issue, open)<br>[#4798](https://github.com/esbmc/esbmc/issues/4798) (issue, open) |
| `regression/quixbugs/shortest_path_length` | error exit | Tracked | [#4778](https://github.com/esbmc/esbmc/issues/4778) (issue, open)<br>[#4799](https://github.com/esbmc/esbmc/issues/4799) (issue, open)<br>+6 more |
| `regression/quixbugs/shortest_path_length_fail` | error exit | Tracked | [#4778](https://github.com/esbmc/esbmc/issues/4778) (issue, open)<br>[#4800](https://github.com/esbmc/esbmc/issues/4800) (issue, open) |
| `regression/z3/quantifiers-field` | error exit | Untracked | — |

### C5a — Unsound: missed bug (false negative) (43)

| Test | Observed | Tracking | Issue / PR |
|---|---|---|---|
| `regression/cbmc/01_cbmc_BV_Arithmetic4` | reports SUCCESSFUL | Untracked | — |
| `regression/cbmc/01_cbmc_Fixedbv23` | reports SUCCESSFUL | Stale tracking | [#1215](https://github.com/esbmc/esbmc/issues/1215) (issue, closed) |
| `regression/cstd/scanf_float_bug` | reports SUCCESSFUL | Untracked | — |
| `regression/cstd/scanf_float_bug_2` | reports SUCCESSFUL | Untracked | — |
| `regression/cstd/strstr_bug` | reports SUCCESSFUL | Untracked | — |
| `regression/cuda/benchmarks/099_test8` | reports SUCCESSFUL | Untracked | — |
| `regression/disabled/ch7_13` | reports SUCCESSFUL | Untracked | — |
| `regression/esbmc-cpp/cpp/ch13_6` | reports SUCCESSFUL | Stale tracking | [#4397](https://github.com/esbmc/esbmc/issues/4397) (issue, closed) |
| `regression/esbmc-cpp/cpp/ch8_9` | reports SUCCESSFUL | Stale tracking | [#4397](https://github.com/esbmc/esbmc/issues/4397) (issue, closed) |
| `regression/esbmc-cpp/cpp/dangling_temporary_ref` | reports SUCCESSFUL | PR provenance only | [#6436](https://github.com/esbmc/esbmc/pull/6436) (PR, closed) |
| `regression/esbmc-cpp/cpp/try-catch_08` | reports SUCCESSFUL | Stale tracking | [#4397](https://github.com/esbmc/esbmc/issues/4397) (issue, closed) |
| `regression/esbmc-cpp/list/list_sort_bug-2` | reports SUCCESSFUL | Tracked | [#4400](https://github.com/esbmc/esbmc/issues/4400) (issue, open)<br>[#6023](https://github.com/esbmc/esbmc/issues/6023) (issue, closed) |
| `regression/esbmc-cpp/map/map_operator_brackets_bug` | reports SUCCESSFUL | Tracked | [#4400](https://github.com/esbmc/esbmc/issues/4400) (issue, open)<br>[#4474](https://github.com/esbmc/esbmc/issues/4474) (issue, closed)<br>+1 more |
| `regression/esbmc-cpp/map/map_string_class-3_bug` | reports SUCCESSFUL | Tracked | [#4400](https://github.com/esbmc/esbmc/issues/4400) (issue, open)<br>[#4473](https://github.com/esbmc/esbmc/issues/4473) (issue, closed)<br>+1 more |
| `regression/esbmc-cpp/template/list_sort_bug` | reports SUCCESSFUL | Tracked | [#4400](https://github.com/esbmc/esbmc/issues/4400) (issue, open)<br>[#4473](https://github.com/esbmc/esbmc/issues/4473) (issue, closed)<br>+2 more |
| `regression/esbmc-solidity/library_11` | reports SUCCESSFUL | Untracked | — |
| `regression/esbmc-solidity/send_ether_via_creation_2` | reports SUCCESSFUL | Untracked | — |
| `regression/esbmc-unix/03_wait_notify` | reports SUCCESSFUL | Untracked | — |
| `regression/esbmc-unix/03_wait_notify2` | reports SUCCESSFUL | Untracked | — |
| `regression/esbmc-unix2/11_cook.fig2.pldi07.extended` | reports SUCCESSFUL | Untracked | — |
| `regression/esbmc-unix2/11_spin2003-cex` | reports SUCCESSFUL | Untracked | — |
| `regression/esbmc-unix2/23_picosat-846_03` | reports SUCCESSFUL | Untracked | — |
| `regression/esbmc-unix2/github_213_mutex_destroy_fail` | reports SUCCESSFUL | Stale tracking | [#213](https://github.com/esbmc/esbmc/issues/213) (issue, closed) |
| `regression/esbmc/force_malloc_success_unrepresentable` | reports SUCCESSFUL | Untracked | — |
| `regression/esbmc/github_159_postdecrement_fail` | reports SUCCESSFUL | Stale tracking | [#159](https://github.com/esbmc/esbmc/issues/159) (issue, closed)<br>[#613](https://github.com/esbmc/esbmc/issues/613) (issue, closed)<br>+2 more |
| `regression/function_contract/github_6483_struct_extent_knownbug` | reports SUCCESSFUL | Stale tracking | [#6483](https://github.com/esbmc/esbmc/issues/6483) (issue, closed)<br>[#6789](https://github.com/esbmc/esbmc/pull/6789) (PR, closed) |
| `regression/function_contract/github_6961_assigns_ptr_var_frame_knownbug` | reports SUCCESSFUL | Stale tracking | [#7067](https://github.com/esbmc/esbmc/issues/7067) (issue, open)<br>[#6961](https://github.com/esbmc/esbmc/issues/6961) (issue, closed)<br>+1 more |
| `regression/function_contract/github_6961_assigns_void_ptr_knownbug` | reports SUCCESSFUL | Stale tracking | [#7067](https://github.com/esbmc/esbmc/issues/7067) (issue, open)<br>[#6961](https://github.com/esbmc/esbmc/issues/6961) (issue, closed)<br>+1 more |
| `regression/function_contract/github_7055_assigns_multilevel_global_knownbug` | reports SUCCESSFUL | Stale tracking | [#7055](https://github.com/esbmc/esbmc/issues/7055) (issue, closed) |
| `regression/function_contract/github_7055_assigns_multilevel_inner_knownbug` | reports SUCCESSFUL | Stale tracking | [#7055](https://github.com/esbmc/esbmc/issues/7055) (issue, closed)<br>[#7068](https://github.com/esbmc/esbmc/pull/7068) (PR, closed) |
| `regression/function_contract/github_7356_requires_global_initialiser_knownbug` | reports SUCCESSFUL | Tracked | [#7356](https://github.com/esbmc/esbmc/issues/7356) (issue, open) |
| `regression/ir-ra/ra-neg-zero-if-merge-lost-sign` | reports SUCCESSFUL | Untracked | — |
| `regression/ir-ra/ra-neg-zero-ternary-order-lost-sign` | reports SUCCESSFUL | Untracked | — |
| `regression/k-induction/github_7565_conditional_entry_fail` | reports SUCCESSFUL | Stale tracking | [#7565](https://github.com/esbmc/esbmc/issues/7565) (issue, closed)<br>[#7587](https://github.com/esbmc/esbmc/pull/7587) (PR, closed) |
| `regression/llvm/compound_literal` | reports SUCCESSFUL | Untracked | — |
| `regression/loop-invariants/6-invariant_in_wrong_place` | reports SUCCESSFUL | Untracked | — |
| `regression/loop-invariants/7-not_conjunct_invariant` | reports SUCCESSFUL | Untracked | — |
| `regression/nonz3/21_printtokens2` | reports SUCCESSFUL | Untracked | — |
| `regression/python/bytes_method_not_str_fail` | reports SUCCESSFUL | PR provenance only | [#5827](https://github.com/esbmc/esbmc/pull/5827) (PR, closed) |
| `regression/python/github_5931_mixed_heap_knownbug_fail` | reports SUCCESSFUL | Stale tracking | [#5931](https://github.com/esbmc/esbmc/issues/5931) (issue, closed) |
| `regression/python/github_7557_localshadow_fail` | reports SUCCESSFUL | Tracked | [#7557](https://github.com/esbmc/esbmc/issues/7557) (issue, open)<br>[#7593](https://github.com/esbmc/esbmc/pull/7593) (PR, closed) |
| `regression/python/github_7674_unresolved_local_import_fail` | reports SUCCESSFUL | Stale tracking | [#7674](https://github.com/esbmc/esbmc/issues/7674) (issue, closed)<br>[#7677](https://github.com/esbmc/esbmc/pull/7677) (PR, closed) |
| `regression/quixbugs/topological_ordering_fail` | reports SUCCESSFUL | Tracked | [#4778](https://github.com/esbmc/esbmc/issues/4778) (issue, open)<br>[#4805](https://github.com/esbmc/esbmc/issues/4805) (issue, open)<br>+6 more |

### C5b — Imprecise: spurious alarm (false positive) (55)

| Test | Observed | Tracking | Issue / PR |
|---|---|---|---|
| `regression/bitwuzla/native_fp_nan_sign` | reports FAILED | Tracked | [#7021](https://github.com/esbmc/esbmc/issues/7021) (issue, open)<br>[#7022](https://github.com/esbmc/esbmc/pull/7022) (PR, closed) |
| `regression/cstd/jump_bug` | reports FAILED | Untracked | — |
| `regression/cstd/scanf-bug-2` | reports FAILED | Untracked | — |
| `regression/cstd/strcoll` | reports FAILED | Untracked | — |
| `regression/cstd/strxfrm` | reports FAILED | Untracked | — |
| `regression/disabled/fstream_rdbuf_1` | reports FAILED | Untracked | — |
| `regression/disabled/humaneval/humaneval_156_disabled` | FAILED after 20 s | Untracked | — |
| `regression/esbmc-cpp/cpp/aggregate_init_named_double_destroy` | reports FAILED | Untracked | — |
| `regression/esbmc-cpp/cpp/aggregate_init_temp_double_destroy` | reports FAILED | Untracked | — |
| `regression/esbmc-cpp/cpp/github_6494_unpaired_new` | reports FAILED | Stale tracking | [#6494](https://github.com/esbmc/esbmc/issues/6494) (issue, closed) |
| `regression/esbmc-cpp/cpp/shared_ptr_member_copy` | reports FAILED | Untracked | — |
| `regression/esbmc-cpp/inheritance/github_7025_vbase_nested_in_base` | reports FAILED | Stale tracking | [#7025](https://github.com/esbmc/esbmc/issues/7025) (issue, closed) |
| `regression/esbmc-cpp/inheritance/github_7025_vbase_shared_diamond` | reports FAILED | Stale tracking | [#7025](https://github.com/esbmc/esbmc/issues/7025) (issue, closed)<br>[#7653](https://github.com/esbmc/esbmc/pull/7653) (PR, closed) |
| `regression/esbmc-cpp/inheritance/vbase_diamond_member_offsets` | reports FAILED | Untracked | — |
| `regression/esbmc-cpp/inheritance/virtual_base_with_base` | reports FAILED | PR provenance only | [#6430](https://github.com/esbmc/esbmc/pull/6430) (PR, closed) |
| `regression/esbmc-cpp11/cpp/allocate_shared_allocator_calls` | reports FAILED | Stale tracking | [#6488](https://github.com/esbmc/esbmc/issues/6488) (issue, closed) |
| `regression/esbmc-solidity/bytes_string_1` | reports FAILED | Untracked | — |
| `regression/esbmc-solidity/github_2564` | reports FAILED | Stale tracking | [#2564](https://github.com/esbmc/esbmc/issues/2564) (issue, closed) |
| `regression/esbmc-solidity/struct_5` | reports FAILED | Untracked | — |
| `regression/esbmc-solidity/type_name_1` | reports FAILED | Untracked | — |
| `regression/esbmc/github_6950-packed-overaligned` | reports FAILED | Stale tracking | [#6950](https://github.com/esbmc/esbmc/issues/6950) (issue, closed)<br>[#6956](https://github.com/esbmc/esbmc/pull/6956) (PR, closed) |
| `regression/esbmc/ptr_rel_huge_object_force_success` | reports FAILED | Untracked | — |
| `regression/function_contract/github_7009_pointer_result_knownbug` | reports FAILED | Tracked | [#7066](https://github.com/esbmc/esbmc/issues/7066) (issue, open)<br>[#7009](https://github.com/esbmc/esbmc/issues/7009) (issue, closed)<br>+1 more |
| `regression/function_contract/github_7056_assigns_large_2d_knownbug` | reports FAILED | Tracked | [#7057](https://github.com/esbmc/esbmc/issues/7057) (issue, open)<br>[#7056](https://github.com/esbmc/esbmc/issues/7056) (issue, closed)<br>+1 more |
| `regression/humaneval/humaneval_1` | reports FAILED | Stale tracking | [#5123](https://github.com/esbmc/esbmc/issues/5123) (issue, closed)<br>[#5124](https://github.com/esbmc/esbmc/issues/5124) (issue, closed)<br>+18 more |
| `regression/humaneval/humaneval_1-1` | reports FAILED | PR provenance only | [#4807](https://github.com/esbmc/esbmc/issues/4807) (issue, closed)<br>[#3497](https://github.com/esbmc/esbmc/pull/3497) (PR, closed)<br>+1 more |
| `regression/humaneval/humaneval_162` | reports FAILED | Untracked | — |
| `regression/humaneval/humaneval_67` | FAILED after 233 s | PR provenance only | [#4807](https://github.com/esbmc/esbmc/issues/4807) (issue, closed)<br>[#4810](https://github.com/esbmc/esbmc/pull/4810) (PR, closed) |
| `regression/humaneval/humaneval_86` | FAILED after 78 s | PR provenance only | [#4807](https://github.com/esbmc/esbmc/issues/4807) (issue, closed)<br>[#4848](https://github.com/esbmc/esbmc/pull/4848) (PR, closed) |
| `regression/humaneval/humaneval_91` | reports FAILED | Weak match only | [#4807](https://github.com/esbmc/esbmc/issues/4807) (issue, closed) |
| `regression/humaneval/humaneval_93` | FAILED after 295 s | Weak match only | [#4807](https://github.com/esbmc/esbmc/issues/4807) (issue, closed)<br>[#5222](https://github.com/esbmc/esbmc/issues/5222) (issue, closed) |
| `regression/humaneval/humaneval_95` | reports FAILED | PR provenance only | [#4807](https://github.com/esbmc/esbmc/issues/4807) (issue, closed)<br>[#4811](https://github.com/esbmc/esbmc/pull/4811) (PR, closed)<br>+1 more |
| `regression/k-induction-parallel/insertion_sort` | reports FAILED | Untracked | — |
| `regression/k-induction/insertion_sort` | reports FAILED | Untracked | — |
| `regression/llvm/indirect_goto` | reports FAILED | Untracked | — |
| `regression/loop-invariants/digital-controller` | reports FAILED | PR provenance only | [#6818](https://github.com/esbmc/esbmc/pull/6818) (PR, closed) |
| `regression/loop-invariants/loop_assigns_large_array_knownbug` | reports FAILED | Stale tracking | [#7103](https://github.com/esbmc/esbmc/issues/7103) (issue, closed)<br>[#7069](https://github.com/esbmc/esbmc/pull/7069) (PR, closed)<br>+1 more |
| `regression/loop-invariants/loop_assigns_moving_index_knownbug` | reports FAILED | PR provenance only | [#7069](https://github.com/esbmc/esbmc/pull/7069) (PR, closed) |
| `regression/numpy/array_return_param_shape_transpose_knownbug` | reports FAILED | PR provenance only | [#7386](https://github.com/esbmc/esbmc/pull/7386) (PR, closed) |
| `regression/numpy/e` | reports FAILED | Untracked | — |
| `regression/numpy/round2` | reports FAILED | Untracked | — |
| `regression/numpy/view_branch_registration_conflict_knownbug` | reports FAILED | Untracked | — |
| `regression/python/github_2848-via-variable` | reports FAILED | Tracked | [#2848](https://github.com/esbmc/esbmc/issues/2848) (issue, open)<br>[#6751](https://github.com/esbmc/esbmc/pull/6751) (PR, closed) |
| `regression/python/github_5931_float_knownbug` | reports FAILED | Stale tracking | [#5931](https://github.com/esbmc/esbmc/issues/5931) (issue, closed) |
| `regression/python/github_5931_str_knownbug` | reports FAILED | Stale tracking | [#5931](https://github.com/esbmc/esbmc/issues/5931) (issue, closed) |
| `regression/python/github_5936_bool_tuple_knownbug` | reports FAILED | Stale tracking | [#5936](https://github.com/esbmc/esbmc/issues/5936) (issue, closed) |
| `regression/python/github_6260_membership_knownbug` | reports FAILED | Stale tracking | [#6260](https://github.com/esbmc/esbmc/issues/6260) (issue, closed) |
| `regression/python/github_7546_inherited` | reports FAILED | Tracked | [#7546](https://github.com/esbmc/esbmc/issues/7546) (issue, open) |
| `regression/python/github_7552_len` | reports FAILED | Tracked | [#7552](https://github.com/esbmc/esbmc/issues/7552) (issue, open) |
| `regression/python/list_eq_nested_comprehension` | reports FAILED | PR provenance only | [#5835](https://github.com/esbmc/esbmc/pull/5835) (PR, closed) |
| `regression/python/set_issubset_knownbug` | reports FAILED | PR provenance only | [#5850](https://github.com/esbmc/esbmc/pull/5850) (PR, closed) |
| `regression/python/string-augmented-repeat` | reports FAILED | Stale tracking | [#4770](https://github.com/esbmc/esbmc/issues/4770) (issue, closed)<br>[#7645](https://github.com/esbmc/esbmc/pull/7645) (PR, closed) |
| `regression/quixbugs/breadth_first_search` | FAILED after 283 s | Tracked | [#4778](https://github.com/esbmc/esbmc/issues/4778) (issue, open)<br>[#4780](https://github.com/esbmc/esbmc/issues/4780) (issue, open)<br>+5 more |
| `regression/quixbugs/rpn_eval` | reports FAILED | Tracked | [#4797](https://github.com/esbmc/esbmc/issues/4797) (issue, open)<br>[#4798](https://github.com/esbmc/esbmc/issues/4798) (issue, open) |
| `regression/quixbugs/topological_ordering` | reports FAILED | Tracked | [#4778](https://github.com/esbmc/esbmc/issues/4778) (issue, open)<br>[#4804](https://github.com/esbmc/esbmc/issues/4804) (issue, open)<br>+6 more |

### C6 — Inconclusive (`VERIFICATION UNKNOWN`) (44)

| Test | Observed | Tracking | Issue / PR |
|---|---|---|---|
| `regression/csmith/csmith05` | reports UNKNOWN | PR provenance only | [#5028](https://github.com/esbmc/esbmc/pull/5028) (PR, closed)<br>[#5029](https://github.com/esbmc/esbmc/pull/5029) (PR, closed) |
| `regression/cvc/quantifiers-sort` | UNKNOWN after 130 s | Untracked | — |
| `regression/cvc/quantifiers-sort-false` | reports UNKNOWN | Untracked | — |
| `regression/disabled/linear_search` | UNKNOWN after 20 s | Weak match only | [#45](https://github.com/esbmc/esbmc/issues/45) (issue, closed)<br>[#306](https://github.com/esbmc/esbmc/issues/306) (issue, closed) |
| `regression/disabled/sum02` | reports UNKNOWN | Untracked | — |
| `regression/disabled/terminator_05` | UNKNOWN after 15 s | Untracked | — |
| `regression/disabled/terminator_06` | UNKNOWN after 20 s | Untracked | — |
| `regression/disabled/trex01` | reports UNKNOWN | Untracked | — |
| `regression/disabled/trex03` | UNKNOWN after 27 s | Untracked | — |
| `regression/k-induction-parallel/check_if` | reports UNKNOWN | Untracked | — |
| `regression/k-induction-parallel/count_down` | reports UNKNOWN | PR provenance only | [#1870](https://github.com/esbmc/esbmc/pull/1870) (PR, closed) |
| `regression/k-induction-parallel/count_down_02` | reports UNKNOWN | PR provenance only | [#1870](https://github.com/esbmc/esbmc/pull/1870) (PR, closed) |
| `regression/k-induction-parallel/count_up_down` | reports UNKNOWN | PR provenance only | [#2651](https://github.com/esbmc/esbmc/pull/2651) (PR, closed)<br>[#3726](https://github.com/esbmc/esbmc/pull/3726) (PR, closed) |
| `regression/k-induction-parallel/for_infinite_loop_2` | reports UNKNOWN | PR provenance only | [#1870](https://github.com/esbmc/esbmc/pull/1870) (PR, closed) |
| `regression/k-induction-parallel/sum01` | reports UNKNOWN | Untracked | — |
| `regression/k-induction-parallel/sum06` | reports UNKNOWN | PR provenance only | [#1870](https://github.com/esbmc/esbmc/pull/1870) (PR, closed) |
| `regression/k-induction-parallel/terminator_04` | reports UNKNOWN | PR provenance only | [#1870](https://github.com/esbmc/esbmc/pull/1870) (PR, closed) |
| `regression/k-induction-parallel/trex02` | reports UNKNOWN | PR provenance only | [#1870](https://github.com/esbmc/esbmc/pull/1870) (PR, closed) |
| `regression/k-induction-parallel/trex03` | reports UNKNOWN | PR provenance only | [#1870](https://github.com/esbmc/esbmc/pull/1870) (PR, closed) |
| `regression/k-induction-parallel/trex04` | reports UNKNOWN | PR provenance only | [#1870](https://github.com/esbmc/esbmc/pull/1870) (PR, closed) |
| `regression/k-induction-parallel/while_infinite_loop_1` | reports UNKNOWN | PR provenance only | [#1870](https://github.com/esbmc/esbmc/pull/1870) (PR, closed) |
| `regression/k-induction-parallel/z_sum_array` | reports UNKNOWN | Untracked | — |
| `regression/k-induction/check_if` | reports UNKNOWN | Untracked | — |
| `regression/k-induction/count_down` | reports UNKNOWN | PR provenance only | [#1870](https://github.com/esbmc/esbmc/pull/1870) (PR, closed) |
| `regression/k-induction/count_down_02` | reports UNKNOWN | PR provenance only | [#1870](https://github.com/esbmc/esbmc/pull/1870) (PR, closed) |
| `regression/k-induction/count_up_down` | reports UNKNOWN | PR provenance only | [#2651](https://github.com/esbmc/esbmc/pull/2651) (PR, closed)<br>[#3726](https://github.com/esbmc/esbmc/pull/3726) (PR, closed) |
| `regression/k-induction/cpp_sum_class` | reports UNKNOWN | PR provenance only | [#4397](https://github.com/esbmc/esbmc/issues/4397) (issue, closed)<br>[#4450](https://github.com/esbmc/esbmc/pull/4450) (PR, closed)<br>+11 more |
| `regression/k-induction/for_infinite_loop_2` | reports UNKNOWN | PR provenance only | [#1870](https://github.com/esbmc/esbmc/pull/1870) (PR, closed) |
| `regression/k-induction/github_1092_1_true` | reports UNKNOWN | Stale tracking | [#1092](https://github.com/esbmc/esbmc/issues/1092) (issue, closed) |
| `regression/k-induction/github_1092_4_true` | reports UNKNOWN | Stale tracking | [#1092](https://github.com/esbmc/esbmc/issues/1092) (issue, closed) |
| `regression/k-induction/github_1092_5_true` | reports UNKNOWN | Stale tracking | [#1092](https://github.com/esbmc/esbmc/issues/1092) (issue, closed) |
| `regression/k-induction/linear_search` | reports UNKNOWN | Weak match only | [#45](https://github.com/esbmc/esbmc/issues/45) (issue, closed)<br>[#306](https://github.com/esbmc/esbmc/issues/306) (issue, closed) |
| `regression/k-induction/sum01` | reports UNKNOWN | Untracked | — |
| `regression/k-induction/sum02` | reports UNKNOWN | Untracked | — |
| `regression/k-induction/sum06` | reports UNKNOWN | PR provenance only | [#5080](https://github.com/esbmc/esbmc/pull/5080) (PR, open)<br>[#1870](https://github.com/esbmc/esbmc/pull/1870) (PR, closed) |
| `regression/k-induction/terminator_04` | reports UNKNOWN | PR provenance only | [#1870](https://github.com/esbmc/esbmc/pull/1870) (PR, closed) |
| `regression/k-induction/terminator_05` | reports UNKNOWN | PR provenance only | [#1870](https://github.com/esbmc/esbmc/pull/1870) (PR, closed) |
| `regression/k-induction/terminator_06` | UNKNOWN after 71 s | PR provenance only | [#1870](https://github.com/esbmc/esbmc/pull/1870) (PR, closed) |
| `regression/k-induction/trex01` | reports UNKNOWN | Untracked | — |
| `regression/k-induction/trex02` | reports UNKNOWN | PR provenance only | [#7454](https://github.com/esbmc/esbmc/pull/7454) (PR, open)<br>[#1193](https://github.com/esbmc/esbmc/pull/1193) (PR, closed)<br>+1 more |
| `regression/k-induction/trex04` | reports UNKNOWN | PR provenance only | [#1870](https://github.com/esbmc/esbmc/pull/1870) (PR, closed) |
| `regression/k-induction/while_infinite_loop_1` | reports UNKNOWN | PR provenance only | [#1870](https://github.com/esbmc/esbmc/pull/1870) (PR, closed) |
| `regression/loop-invariants/cpp_sum_class` | reports UNKNOWN | PR provenance only | [#4397](https://github.com/esbmc/esbmc/issues/4397) (issue, closed)<br>[#4450](https://github.com/esbmc/esbmc/pull/4450) (PR, closed)<br>+11 more |
| `regression/mopsa/object_in_loop` | UNKNOWN after 216 s | PR provenance only | [#5913](https://github.com/esbmc/esbmc/pull/5913) (PR, closed) |

### C7 — No answer within budget (performance) (16)

| Test | Observed | Tracking | Issue / PR |
|---|---|---|---|
| `regression/esbmc-cpp/cpp/istringstream_extract` | no answer in 300 s | PR provenance only | [#6409](https://github.com/esbmc/esbmc/pull/6409) (PR, closed) |
| `regression/esbmc-unix/github_2513_6` | no answer in 300 s | Tracked | [#3459](https://github.com/esbmc/esbmc/issues/3459) (issue, open)<br>[#2513](https://github.com/esbmc/esbmc/issues/2513) (issue, closed)<br>+5 more |
| `regression/floats-regression/remquo` | no answer in 300 s | Untracked | — |
| `regression/humaneval/humaneval_90` | no answer in 300 s | Weak match only | [#4807](https://github.com/esbmc/esbmc/issues/4807) (issue, closed) |
| `regression/mopsa/list_in_loop` | no answer in 300 s | PR provenance only | [#5929](https://github.com/esbmc/esbmc/pull/5929) (PR, closed) |
| `regression/python/concurrency_fail` | no answer in 300 s | Tracked | [#4566](https://github.com/esbmc/esbmc/issues/4566) (issue, open)<br>[#4568](https://github.com/esbmc/esbmc/issues/4568) (issue, closed)<br>+5 more |
| `regression/quixbugs/depth_first_search` | no answer in 300 s | Stale tracking | [#4778](https://github.com/esbmc/esbmc/issues/4778) (issue, open)<br>[#4782](https://github.com/esbmc/esbmc/issues/4782) (issue, closed)<br>+8 more |
| `regression/quixbugs/flatten_fail` | no answer in 300 s | Tracked | [#4778](https://github.com/esbmc/esbmc/issues/4778) (issue, open)<br>[#4786](https://github.com/esbmc/esbmc/issues/4786) (issue, open)<br>+1 more |
| `regression/quixbugs/knapsack` | no answer in 300 s | Tracked | [#4789](https://github.com/esbmc/esbmc/issues/4789) (issue, open)<br>[#3811](https://github.com/esbmc/esbmc/issues/3811) (issue, closed) |
| `regression/quixbugs/knapsack_fail` | no answer in 300 s | Tracked | [#4778](https://github.com/esbmc/esbmc/issues/4778) (issue, open)<br>[#4789](https://github.com/esbmc/esbmc/issues/4789) (issue, open)<br>+1 more |
| `regression/quixbugs/powerset` | no answer in 300 s | Tracked | [#4794](https://github.com/esbmc/esbmc/issues/4794) (issue, open)<br>[#4795](https://github.com/esbmc/esbmc/issues/4795) (issue, open) |
| `regression/quixbugs/powerset_fail` | no answer in 300 s | Tracked | [#4778](https://github.com/esbmc/esbmc/issues/4778) (issue, open)<br>[#4795](https://github.com/esbmc/esbmc/issues/4795) (issue, open)<br>+1 more |
| `regression/quixbugs/shortest_path_lengths` | no answer in 300 s | Stale tracking | [#4778](https://github.com/esbmc/esbmc/issues/4778) (issue, open)<br>[#3915](https://github.com/esbmc/esbmc/issues/3915) (issue, closed)<br>+4 more |
| `regression/quixbugs/shortest_path_lengths_fail` | no answer in 300 s | Stale tracking | [#4778](https://github.com/esbmc/esbmc/issues/4778) (issue, open)<br>[#4801](https://github.com/esbmc/esbmc/issues/4801) (issue, closed) |
| `regression/quixbugs/shortest_paths` | no answer in 300 s | Tracked | [#4778](https://github.com/esbmc/esbmc/issues/4778) (issue, open)<br>[#4802](https://github.com/esbmc/esbmc/issues/4802) (issue, open)<br>+11 more |
| `regression/quixbugs/shortest_paths_fail` | no answer in 300 s | Tracked | [#4778](https://github.com/esbmc/esbmc/issues/4778) (issue, open)<br>[#5444](https://github.com/esbmc/esbmc/issues/5444) (issue, open)<br>+2 more |

### C8 — Other output mismatch (6)

| Test | Observed | Tracking | Issue / PR |
|---|---|---|---|
| `regression/cstd/printf_bug_1` | reports FAILED | Untracked | — |
| `regression/cstd/printf_bug_2` | reports FAILED | Untracked | — |
| `regression/cstd/printf_bug_3` | reports FAILED | Untracked | — |
| `regression/esbmc/github_2220_vla_bound` | no verdict | Stale tracking | [#2220](https://github.com/esbmc/esbmc/issues/2220) (issue, closed)<br>[#6994](https://github.com/esbmc/esbmc/pull/6994) (PR, closed) |
| `regression/esbmc/linking-7` | reports FAILED | Untracked | — |
| `regression/goto-coverage/github_1720_6` | no verdict | PR provenance only | [#1720](https://github.com/esbmc/esbmc/pull/1720) (PR, closed) |

## 9. Reproducing this

```sh
# the population
grep -rl '^KNOWNBUG' regression --include=test.desc | sed 's:/test.desc::'

# one test's actual behaviour, exactly as the harness runs it
sed -n '2,3p' regression/<suite>/<test>/test.desc   # source file, then flags

# whether a bug is already fixed
ctest -R 'regression/<suite>/<test>$' --output-on-failure
#   Passed  -> the bug is still live
#   Failed with 'passed but is marked as KNOWNBUG' -> it is fixed; reclassify to CORE
```

Issue linkage was built from `gh api 'repos/esbmc/esbmc/issues?state=all&per_page=100'
--paginate`, matched by declared reference, then by full test path, then by bare name.

