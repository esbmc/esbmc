# Plan — affording the schedule space #6607 exposed (issue #6831, cause 1)

**Status:** W0 shipped. W2 diagnosed and re-scoped — `--state-hashing` is
**unsound**, not merely useless (see W2). W1, W3, W4 not started.
**Owner issue:** [#6831](https://github.com/esbmc/esbmc/issues/6831), *cause 1 —
schedule-space explosion*, 291 of 489 lost SV-COMP tasks.
**Bisected to:** `bac652b13c` — `[goto-symex] Track main-thread termination per
state, not per search` (#6607), which fixes #4584.
**Last updated:** 2026-08-09.

**Measurement environment.** All numbers below were measured on an x86_64 Linux
host against `build/src/esbmc/esbmc`, ESBMC 8.4.0, built from `19db2adc96`
(a descendant of `bac652b13c`, so the post-#6607 behaviour). Solver: Bitwuzla
0.9.0 (the default). Single run per configuration, one machine, no repetition —
treat the ratios as indicative and the orderings as reliable, not the absolute
seconds. Reproducer: `regression/esbmc-unix/01_malloc_20`, the same one the
issue's bisect used.

`master` has since advanced to `8ffb84b24d`, which includes #6793
(`malloc(0)` may now return NULL or a freeable object). That commit is *not* in
the measurement binary and it touches allocation modelling in a reproducer
named `01_malloc_20` — re-measure §2 against current `master` before acting on
the absolute numbers. The relative standings (MPOR prunes, hashing does not;
symex dominates the solver) are not expected to move.

---

## 1. Premise: #6607 is not the defect

`check_if_ileaves_blocked()` consulted `art1->main_thread_ended`, a
reachability-tree flag set when `__ESBMC_main` ends and never cleared on
backtracking. The first branch to run main to completion disabled interleaving
generation for every branch explored afterwards. #6607 made the check per-state
(`threads_state[0].thread_ended`).

The pre-#6607 interleaving counts were the *unsound* ones. **Reverting #6607 is
a non-goal of this plan** (§8). The question this document answers is how to
afford the schedule space ESBMC now correctly explores.

One scoping note the issue does not make explicit: the guard is skipped
entirely when `deadlock-check` or `data-races-check` is set
(`execution_state.cpp:517-521`), both before and after #6607. So the
`no-data-race` category never went through this path — consistent with the lost
tasks being dominated by `pthread-wmm` under `unreach-call`.

---

## 2. Measurements

### 2.1 What the existing reduction knobs are worth

`regression/esbmc-unix/01_malloc_20` with its own `test.desc` flags
(`--no-unwinding-assertions --unwind 3 --context-bound 2 --force-malloc-success
--memory-leak-check`):

| configuration | interleavings | wall | verdict |
|---|---|---|---|
| baseline (MPOR on, hashing off) | 939 | 25.2 s | SUCCESSFUL |
| `--state-hashing` | **939** | 26.1 s | SUCCESSFUL |
| `--no-por` | 1170 | 36.8 s | SUCCESSFUL |
| `--no-por --state-hashing` | 1154 | 37.3 s | SUCCESSFUL |

Two results drive this plan:

- **MPOR prunes ~20 %** (1170 → 939). It works, but it is the only thing working.
- **State hashing prunes nothing at all** (939 → 939) and costs ~4 % wall.

The 939 reproduces the issue's reported 940.

**Superseded on the hashing row.** W2 re-measured this across all 314
concurrent CORE tests rather than this one: hashing prunes on 62 of them and
saves 16.5 % wall overall. `01_malloc_20` is an outlier, not a summary. It is
also unsound on four of them — see W2.

### 2.2 The same reproducer under the SV-COMP flag shape

`scripts/competitions/svcomp/esbmc-wrapper.py` passes **no `--context-bound`**,
so `CS_bound == -1` and `check_for_hash_collision()` takes its unconditional
branch (`reachability_tree.cpp:460`) rather than the budget-gated one. Dropping
`--context-bound 2`, 300 s cap:

| configuration | interleavings reached | wall | verdict |
|---|---|---|---|
| no context bound | 31,423 | 300 s | **TIMEOUT** |
| no context bound, `--state-hashing` | 30,872 | 300 s | **TIMEOUT** |

So state hashing prunes **1.8 %** even on its more aggressive path. The wrapper
already passes `--state-hashing` in `esbmc_dargs`. The issue's suggested action
"state hashing on by default for concurrency" is therefore **already in effect
and is not the lever** — this plan's §4 W2 proposes fixing or dropping it.

Note the 33× gap between the bounded and unbounded runs. The schedule space is
not merely large; without a context bound it is not enumerable at all here.

### 2.3 Where the time goes

Summed over the 940 completed formulas of the bounded baseline run:

| phase | calls | total |
|---|---|---|
| symex | 940 | **16.815 s** |
| slicing | 940 | 0.813 s |
| encoding to solver | 779 | 1.590 s |
| decision procedure | 779 | **1.753 s** |
| BMC program time | 940 | 22.710 s |

**Symex outweighs the solver by ~10×.** 290,165 VCCs are generated across the
940 schedules; a single schedule generates 376. This is a schedule-enumeration
cost, not an SMT cost, and the plan is scoped accordingly: work aimed at the
solver would be aimed at 7.7 % of the run.

---

## 3. Diagnosis

Two orthogonal levers, both open:

- **A — explore fewer schedules.** MPOR is the only reduction that fires on this
  reproducer; W2 shows state hashing fires on plenty of others, unsoundly.
  W0 added the counters this diagnosis needed.
- **B — make each schedule cheaper.** Each of the 940 schedules is symexed,
  sliced, encoded and solved as an independent formula. The DFS restores
  execution states on backtracking, but the per-formula pipeline downstream of
  symex does not exploit the shared prefix.

---

## 4. Workstreams

### W0 — Instrument the reduction (prerequisite, no behaviour change) — **done**

Nothing reported schedules pruned by MPOR, pruned by state hashing, pruned by
the context bound, or explored. Every measurement in §2 had to be obtained by
toggling flags and re-running whole verifications, which cannot be done across
an SV-COMP set.

`reachability_treet::reduction_stats` now carries `peak_threads`,
`schedules_explored`, `pruned_by_mpor`, `pruned_by_hash` and
`pruned_by_cs_bound`, emitted as one line at the end of a run:

```
Schedule reduction: peak_threads 3, schedules_explored 940, pruned_by_mpor 296, pruned_by_hash 0, pruned_by_cs_bound 335643
```

The three prune counters share a unit — context-switch points at which that
reduction stopped the search branching further — while `schedules_explored`
counts formulas. The line is suppressed when `peak_threads == 1`, so sequential
runs are unaffected. `peak_threads` is what makes that gate possible; it is not
otherwise part of the plan.

**Exit — discharged.** §2.1 now reads off one run per configuration:

| configuration | schedules | pruned_by_mpor | pruned_by_hash |
|---|---|---|---|
| baseline (MPOR on, hashing off) | 940 | 296 | 0 |
| `--state-hashing` | 940 | 296 | **0** |
| `--no-por` | 1171 | 0 | 0 |
| `--no-por --state-hashing` | 1155 | 0 | 20 |

(One more than §2.1's counts: that table quoted `interleaving_number`, which is
read before the final interleaving increments it.) Hashing is now shown to
prune nothing *behind MPOR* and only 20 of 1175 with MPOR off — the two
reductions are not additive, and W2's hypothesis can be tested without a
rebuild.

### W1 — Make partial-order reduction prune harder (highest leverage)

`calculate_mpor_constraints()` (`execution_state.cpp:1232`) implements the
MPOR quasi-monotonic dependency-chain test, with dependencies derived from
`thread_last_reads` / `thread_last_writes` per transition. It is the only
reduction that fires, and it leaves 939 schedules where the property under test
touches a handful of shared objects.

Candidates, cheapest first:

1. **Sleep sets** layered on the existing MPOR. Classical, sound, composes with
   a persistent-set-style reduction rather than replacing it; small state per
   DFS node.
2. **Sharpen the independence relation.** Dependencies are currently computed
   from the last transition's read/write sets. Objects proved thread-local by
   the existing value-set / pointer analysis can never induce a dependency;
   `--cswitch-skip-readonly-globals` (already passed by the SV-COMP wrapper)
   shows the shape of this argument but applies it only to read-only globals.
3. **Evaluate DPOR** (Flanagan–Godefroid dynamic POR) against MPOR. This is a
   design change, not a patch, and is scoped as investigation only until 1 and 2
   are measured.

**Exit:** ≥2× reduction in schedules explored on `01_malloc_20` at
`--context-bound 2`, with no verdict change anywhere in the concurrency
regression suites, and #4584's regression test still detecting its race.

### W2 — Fix state hashing or stop paying for it — **re-scoped: it is unsound**

W2 was written around "prunes ~0 for ~4 % wall". Both halves of that premise
were artefacts of measuring one reproducer. Running every CORE test in
`esbmc-unix` and `esbmc-unix2` twice — once stock, once with `--state-hashing`
— over the 314 that reach more than one thread, with W0's counters:

- **Verdict changes: 4**, every one of them `FAILED` → `SUCCESSFUL`.
- Schedules: fewer with hashing on **62** tests, equal on 247, more on none.
- Wall over the tests whose verdict agreed: 372 s → 310 s, **−16.5 %**.

So hashing does prune, sometimes enormously (`11_cook.fig2.pldi07`
11,130 → 79 schedules, 25.0 s → 0.6 s; `github_3449` 2134 → 621). `01_malloc_20`,
where §2.1 measured it, is simply a case where it prunes nothing — it is not
representative.

**The finding that matters is the soundness one.** These four CORE tests, whose
`test.desc` expects `VERIFICATION FAILED`, return `VERIFICATION SUCCESSFUL`
when `--state-hashing` is added, on stock `master` with no other flag change:

| test | stock | `--state-hashing` |
|---|---|---|
| `esbmc-unix2/00_mpor1` | FAILED | **SUCCESSFUL** |
| `esbmc-unix2/00_mpor2` | FAILED | **SUCCESSFUL** |
| `esbmc-unix/race_guard_other_write` | FAILED | **SUCCESSFUL** |
| `esbmc-unix/race_guard_self_clear` | FAILED | **SUCCESSFUL** |

The fingerprint collides two states that are not bisimilar, and
`post_hash_collision_cleanup()` then marks every switch from that point
explored, so the schedule carrying the violation is never generated. This is
the §7 "re-truncation" failure mode, already shipped, and
`esbmc-wrapper.py:237` passes `--state-hashing` in `esbmc_dargs` — so SV-COMP
runs are exposed to it. A wrong `true` is scored far more harshly than an
`unknown`, which makes this a bigger liability than any timeout W1–W3 address.

Pinned by `regression/esbmc-unix/github_6831_hashing_unsound{,_mpor}`
(KNOWNBUG, expecting the correct FAILED) so a fix flips them to CORE.

Revised order:

1. **Diagnose the collision.** The hypothesis that the fingerprint is too
   *fine* is now the wrong end: it is too *coarse* somewhere. Not yet located —
   an allocation-naming hypothesis (the monotone `dynamic_counter` making
   post-`malloc` states unique) was tested with a controlled probe and
   **rejected**: the probe still pruned 76 states.
2. **Fix or disable.** Until a fix exists, disabling `--state-hashing` in
   `esbmc_dargs` is the sound choice, and costs the −16.5 % above.
3. Only then revisit making it prune harder.

**Exit:** the four tests above pass with `--state-hashing`, or the flag is out
of `esbmc_dargs`. Performance tuning of the fingerprint is not in scope until
one of those holds.

### W3 — Stop redoing per-schedule work

940 schedules produce 940 slices, 779 encodings, 779 solver calls and 290,165
VCCs from a program that yields 376 VCCs on one schedule. Sibling schedules
share a long common prefix by construction (they differ only after the
backtrack point).

Investigate, in this order: incremental solving across siblings (push/pop over
the shared prefix, which the incremental-BMC path SV-COMP uses already has
machinery for); and not re-slicing the prefix. This is lever B and is
independent of W1/W2 — it reduces the constant, not the exponent, but §2.3 says
the constant is where 74 % of the time is.

**Exit:** measurable wall-time reduction on `01_malloc_20` at an unchanged
schedule count and unchanged verdicts.

### W4 — Bound the schedule space in the SV-COMP strategy

§2.2 shows the wrapper's own configuration cannot enumerate this reproducer at
all. `--incremental-context-bound` exists (`options.cpp:553`, "stops at the
first violation or once a round has covered every interleaving") and the
wrapper does not use it.

Proposal: for the concurrency categories, explore shallow schedules first and
deepen while the budget lasts, so a task that currently times out with no answer
instead answers at the largest bound it can afford. **Soundness constraint that
must be honoured:** a `true` verdict may only be emitted when a round has
provably covered every interleaving — a `true` from an exhausted-budget bound is
an unsound claim and must be reported `unknown`. Verify this against the
existing implementation before proposing any wrapper change; if the flag does
not already guarantee it, W4 becomes a code change, not a configuration change.

**Exit:** a measured score delta on the affected categories, and an explicit
argument for why no `true` is emitted without exhaustive coverage.

---

## 5. Sequencing

W0 first — W1 and W2 cannot be assessed without it. W1 and W2 are then
independent and can proceed in parallel; W3 is independent of both. W4 is
gated on W0 (to size the win) but not on W1–W3.

W1 has the highest expected value and the highest risk; W2 has the smallest
scope and a guaranteed outcome (a fix or a removal). W2 is the recommended
first substantive PR after W0.

---

## 6. Gates

Any PR under W1, W2 or W4 changes which schedules are explored, i.e. it can
re-introduce exactly the unsoundness #6607 removed. Each must discharge:

- **G1 — no verdict change.** Full `esbmc-unix` and concurrency regression
  suites, before and after, verdict-for-verdict identical.
- **G2 — #4584 still caught.** The regression test #6607 added still detects its
  race. A reduction that silently re-truncates the search will pass G1 and fail
  only here.
- **G3 — schedule count pinned.** A regression test asserting the interleaving
  count on `01_malloc_20` at `--context-bound 2`, so a future truncation is a
  test failure rather than a score movement noticed a release later. This is the
  oracle the issue's bisect used; it should be in the tree.
- **G4 — dual-solver agreement.** Bitwuzla and Z3 agree on the changed set.
- **G5 — measured, not asserted.** Every claimed reduction quoted with W0's
  counters, before and after, naming the configuration.

A reduction that cannot discharge G2 is unsound and is not shipped, whatever it
does to the score.

---

## 7. Risks

- **Re-truncation.** The failure mode of this entire plan is a reduction that
  looks like a speedup and is a silent loss of coverage. G2 and G3 exist for
  this and are not optional.
- **DPOR is a rewrite.** W1.3 is scoped as investigation deliberately; it should
  not be started before W1.1/W1.2 are measured.
- **Single-machine measurement.** §2 is one host, one run per configuration.
  Before acting on a ratio, re-measure.
- **W4 can trade soundness for score** if the coverage constraint is not
  honoured. Called out in W4 rather than left implicit.

---

## 8. Non-goals

- **Reverting #6607.** It fixes a real unsoundness (#4584).
- **Cause 2 of #6831** (the ~3.5 % general slowdown plus ~0.15 s fixed cost,
  ~198 lost tasks, led by `Juliet_Test` at ~99 s of a 100 s limit). It is a
  separate defect with a separate fix — on-demand operational-model loading —
  and is paid by all 36,603 tasks rather than by the concurrency categories.
  It deserves its own plan.
- **The `incorrect 6 → 8` delta.** The issue establishes it came from the
  benchmark repository moving between runs, not from ESBMC.
- **Pinning the sv-benchmarks revision** (the issue's suggested action 4). A
  benchmarking-infrastructure change, unrelated to the schedule space.

---

## 9. One-line summary

The solver is not the problem: 74 % of the run is symex over 940 schedules,
MPOR prunes 20 % of them, state hashing prunes none, and nothing reports either
— so instrument the reduction (W0), then make it prune (W1/W2) and make each
schedule cheaper (W3), never by exploring fewer schedules than #6607 proved are
reachable.
