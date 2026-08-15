# Plan — affording the schedule space #6607 exposed (issue #6831, cause 1)

**Status:** W0, W2, W1.1 and W4.1 shipped; `--state-hashing` was **unsound** and
is fixed (see W2); `--sleep-sets` is new, off by default, and sound on the paths
it is allowed to run on (see W1.1). W3 and W4 are investigated, and both turn
out to be about existing machinery rather than new: W3's exit is already
discharged by `--smt-during-symex`, and W4 is re-scoped from a wrapper change to
a code change. W3's remaining wrapper question is now closed too (W3.3) — the
flag was already on, and the segfault that closing it uncovered on the shipped
configuration is fixed.
**Owner issue:** [#6831](https://github.com/esbmc/esbmc/issues/6831), *cause 1 —
schedule-space explosion*, 291 of 489 lost SV-COMP tasks.
**Bisected to:** `bac652b13c` — `[goto-symex] Track main-thread termination per
state, not per search` (#6607), which fixes #4584.
**Last updated:** 2026-08-15.

**Measurement environment.** Except where a workstream names its own build (W3.3
does), all numbers below were measured on an x86_64 Linux host against
`build/src/esbmc/esbmc`, ESBMC 8.4.0, built from `19db2adc96`
(a descendant of `bac652b13c`, so the post-#6607 behaviour). Solver: Bitwuzla
0.9.0 (the default). Single run per configuration, one machine, no repetition —
treat the ratios as indicative and the orderings as reliable, not the absolute
seconds. Reproducer: `regression/esbmc-unix/01_malloc_20`, the same one the
issue's bisect used.

`master` has since advanced past `8ffb84b24d`, which includes #6793
(`malloc(0)` may now return NULL or a freeable object) and touches allocation
modelling in a reproducer named `01_malloc_20`, so §2 was flagged for
re-measurement before its absolute numbers were acted on. **Done — §2.3 is
re-measured below and the standings hold.** Every phase came in 8–10 % faster
with the call counts (940/940/779/779) and the reduction counters
(940 schedules, 296 MPOR, 0 hash) byte-identical, so the ratios §2.3 rests on
are unchanged.

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
concurrent CORE tests rather than this one: hashing cuts schedules on 41 of
them and saves ~12 % wall overall once its soundness bug is fixed.
`01_malloc_20` is an outlier, not a summary — see W2.

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

Summed over the 940 completed formulas of the bounded baseline run, as first
measured and as re-measured on current `master` (2026-08-10):

| phase | calls | total | re-measured |
|---|---|---|---|
| symex | 940 | **16.815 s** | **15.408 s** |
| slicing | 940 | 0.813 s | 0.699 s |
| encoding to solver | 779 | 1.590 s | 1.387 s |
| decision procedure | 779 | **1.753 s** | **1.603 s** |
| BMC program time | 940 | 22.710 s | 20.457 s |

**Symex outweighs the solver by ~10×.** 290,165 VCCs are generated across the
940 schedules; a single schedule generates 376. This is a schedule-enumeration
cost, not an SMT cost, and the plan is scoped accordingly: work aimed at the
solver would be aimed at 7.7 % of the run.

---

## 3. Diagnosis

Two orthogonal levers, both open:

- **A — explore fewer schedules.** MPOR is the only reduction that fires on this
  reproducer; W2 shows state hashing fires on plenty of others, unsoundly.
  W0 added the counters this diagnosis needed. W1.1 adds a third reduction,
  sleep sets, which fires only where the search is exhaustive — so it is worth
  nothing under a context bound and a good deal without one.
- **B — make each schedule cheaper.** Each of the 940 schedules is symexed,
  sliced, encoded and solved as an independent formula. The DFS restores
  execution states on backtracking, but the per-formula pipeline downstream of
  symex does not exploit the shared prefix. **Closed by W3:** `--smt-during-symex`
  already makes it exploit the prefix, for −13.5 % (−7.2 % under the wrapper's
  own flags, W3.3), and what remains under this lever is ~5 % of the run. Lever A
  is the only one with headroom left.

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

#### What actually drives the dependencies (measured)

Instrumenting `mpor_set_conflicts` to name the object behind every conflict,
and histogramming over five benchmarks:

| benchmark | top dependency drivers |
|---|---|
| `01_malloc_20` | the mutex + 2 condvars (996), pthread bookkeeping (260), user `num` (**1**) |
| `11_cook.fig2.pldi07` | `__ESBMC_pthread_thread_ended` (4591), user `x` (4367) |
| `github_3449` | user mutex `m` (6181), user `counter` (608), bookkeeping (437) |
| `11_bakery.simple.preempt` | users `t2`, `t1`, `x` (337) — all user |
| `github_6475_safe` | `__ESBMC_pthread_thread_ended` (2172), user lock `l` (1960), `__ESBMC_pthread_join_waiters` (654) |

Two things follow. Dependencies are dominated by **synchronisation objects and
pthread-model bookkeeping**, not by ordinary shared data — on `01_malloc_20`
exactly one conflict in 1257 involves a user global. And the mix varies enough
between benchmarks that no single-reproducer measurement should be trusted
(the mistake §2.1 made).

#### W1.2 was tried and does not work — do not re-try it

`mpor_keys_may_alias` already keys pthread *sync* arrays per element, but
`__ESBMC_pthread_thread_ended` (`_Bool[]`), `__ESBMC_pthread_join_waiters`
(`unsigned[]`) and `__ESBMC_pthread_end_values` (`void *[]`) are thread-indexed
bookkeeping that fell back to a whole-array key, so every thread's write to its
own slot conflicted with every other thread's read of a different slot.
Extending element keying to those arrays is sound and precise — and bought
**nothing**: the conflict count moved intact to the next bookkeeping object
(`thread_ended` → `end_values` → `__ESBMC_num_threads_running`), which is a
*scalar* and cannot be refined at all. Schedule counts were byte-identical on
all 7 benchmarks tested.

The dependency is over-determined: thread start/end threads every pair of
transitions through shared scalar bookkeeping, so no amount of aliasing
precision separates them. **The independence relation is not the lever.**

Remaining candidates:

1. **Sleep sets** layered on the existing MPOR. Classical, sound, composes with
   a persistent-set-style reduction rather than replacing it; small state per
   DFS node. Now the first thing to try, not the second. **Done — W1.1 below.**
2. ~~**Decouple the pthread model's bookkeeping.**~~ **Closed — measured below
   (W1.3), its ceiling is 3.5 % on the exit benchmark.**
3. **Evaluate DPOR** (Flanagan–Godefroid dynamic POR) against MPOR. A design
   change, not a patch; investigation only. **Done — W1.4 below: do not start
   it.**

#### W1.3 — Decoupling the bookkeeping cannot reach the exit (measured)

Candidate 2 was to make `__ESBMC_num_threads_running` and friends per-thread so
W1.2's element keying would start paying. Rather than build the operational
model first, its **ceiling** was measured: `mpor_set_conflicts` was patched to
drop every conflict whose key names a `__ESBMC_pthread*` /
`__ESBMC_num_threads_running` / `__ESBMC_blocked_threads_count` object. That is
unsound — it is an upper bound on what *any* decoupling of that state could
buy, not a candidate patch, and it was reverted rather than committed.

| benchmark | schedules, base | bookkeeping conflicts dropped | change |
|---|---|---|---|
| `01_malloc_20` (`--context-bound 2`) | 940 | 907 | **−3.5 %** |
| `11_cook.fig2.pldi07` | 11,130 | 11,124 | −0.05 % |
| `github_3449` | 2134 | 1606 | −24.7 % |
| `11_bakery.simple.preempt` | 2279 | 2279 | 0 % |
| `github_6475_safe` | 2544 | 2348 | −7.7 % |

The exit asks for ≥2× on `01_malloc_20`; the unsound ceiling delivers 3.5 %.
**Candidate 2 is closed** — the operational-model work it needs cannot pay for
itself.

**The methodological finding is the more useful one: conflict-count share does
not predict schedule-count reduction.** The histogram above ranks
`__ESBMC_pthread_thread_ended` as the top dependency driver on
`11_cook.fig2.pldi07` with 4591 conflicts — and removing every bookkeeping
conflict there changes the schedule count by six. The dependencies are
over-determined in the same way W1.2 found: drop one driver and the remaining
ones still order the same pairs of transitions. Rank a lever by re-measuring
schedules with it disabled, never by how often it appears in a conflict
histogram.

#### W1.4 — What is left for DPOR, measured (candidate 3: do not start it)

DPOR computes the dependency relation during exploration rather than from a
static analysis (Flanagan and Godefroid, *"Dynamic Partial-Order Reduction for
Model Checking Software"*, POPL 2005,
[doi:10.1145/1040305.1040315](https://doi.org/10.1145/1040305.1040315)), and its
optimal form explores exactly one interleaving per Mazurkiewicz trace — the
lower bound any sound partial-order reduction can reach (Abdulla, Aronis,
Jonsson and Sagonas, *"Optimal Dynamic Partial Order Reduction"*, POPL 2014,
[doi:10.1145/2535838.2535845](https://doi.org/10.1145/2535838.2535845)). So the
question DPOR answers is: **how much of what ESBMC explores is redundant that
its existing reductions do not already remove?** Measured, `schedules_explored`
per configuration:

| benchmark | `--no-por` | MPOR | + sleep | + hash | + both |
|---|---|---|---|---|---|
| `01_malloc_20` (`--context-bound 2`) | 1171 | 940 | 940 | 940 | 940 |
| `github_3449` | TIMEOUT | 2134 | 2131 | 876 | 846 |
| `11_bakery.simple.preempt` | TIMEOUT | 2279 | 2279 | 746 | 746 |

**The exit benchmark has no redundancy left to find.** On `01_malloc_20` at
`--context-bound 2`, every reduction ESBMC has — MPOR, sleep sets, state
hashing, all combinations — lands on exactly 940. W1's exit asks for ≥2× *on
this benchmark*, and no technique available today moves it at all. **Re-scope
the exit** onto a benchmark where redundancy is demonstrable (`github_3449`,
`11_bakery.simple.preempt`); keeping it on `01_malloc_20` makes W1 unfalsifiable
rather than demanding.

**The reductions largely substitute for each other.** On these three benchmarks
sleep sets add almost nothing on top of MPOR (2134 → 2131, 2279 → 2279) while
being strong *instead* of it: with `--no-por`, `github_3449` goes from TIMEOUT
to **1653**, beating MPOR's 2134, and `github_6831_sleep_sets` goes 4892 → 368.
Read that alongside W1.1's larger sample rather than instead of it — over 331
tests it found `--sleep-sets` behind MPOR still cuts 63 of them, so "adds
nothing behind MPOR" is true of this trio, not of the corpus. What the trio does
show is that two reductions each remove much of the *same* redundancy, which is
the case against DPOR here: it is a rewrite (§7), and neither this trio nor
W1.1's sweep exhibits the residue that would pay for one.

**Counter caveat, and it matters for reading every row above.** An MPOR or hash
prune cuts the prefix and `get_next_formula` still returns a formula for it, so
`schedules_explored` counts that prefix as a schedule. Under DFS the complete
schedules are exactly

```
schedules_explored - pruned_by_mpor - pruned_by_hash
```

which was checked against every configuration measured here (e.g. `github_3449`
under `--state-hashing`: 1107 formulas, 329 MPOR + 489 hash, and 818 = 329+489).
The correction is large enough to invert a reading: `--no-por --state-hashing`
on `github_3449` reports 15,923 formulas against 1653 for `--no-por
--sleep-sets`, which looks like a 9.6× loss and is really 4596 vs 1653, a 2.8×
one. Subtract before comparing configurations.

A separate `truncated_formulas` counter was implemented for this and then
**dropped**: under DFS it always equalled `pruned_by_mpor + pruned_by_hash`, so
it carried no information the line did not already have, at the cost of eleven
`test.desc` updates. It differs only under `--schedule`, where the whole metric
degenerates to 1.

**Recommendation:** close candidate 3 without building anything. Revisit only if
a benchmark appears where the schedule count stays high under MPOR, sleep sets
*and* hashing — that is the shape whose redundancy only a trace-based method
could remove, and this corpus does not contain one.

#### W1 exit — re-scoped

The original exit ("≥2× reduction in schedules explored on `01_malloc_20` at
`--context-bound 2`") is not a bar this workstream can clear or fail on the
merits: W1.1 found it names the one configuration in which sleep sets cannot
fire, and W1.4 finds no reduction ESBMC has moves it at all. Replace it with:

- **A measured reduction on a benchmark that has redundancy to remove**, quoted
  as complete schedules (`schedules_explored - pruned_by_mpor - pruned_by_hash`)
  against the same benchmark's best existing configuration. `github_3449` (876
  under MPOR + hashing) and `11_bakery.simple.preempt` (746) are the current
  bars; `01_malloc_20` at `--context-bound 2` is retired as an exit benchmark
  and kept only as the bisect oracle it already serves as (gate G3).
- **No verdict change** across the 343 CORE concurrent tests, unchanged.
- **#4584's regression tests still detecting their race**, unchanged.

Nothing about the soundness gates changes; what changes is that the performance
bar now points at a program where the quantity it measures can move.

#### W1.1 — Sleep sets (`--sleep-sets`), shipped off by default

They do not layer on MPOR, and the exit criterion above was unreachable as
written. Both corrections come from the same fact: **a sleep set may only
record a thread whose subtree was exhaustively explored**, and under
`--context-bound 2` — the configuration the exit criterion names — almost
nothing is.

Six independent unsoundnesses were found on the way, each of which produced a
plausible-looking speedup while silently dropping violating schedules. They are
recorded because each is easy to reintroduce. (d) and (e) were found by code
review after (a)–(c) were already fixed and the suite was green, which is the
strongest argument in this document for reviewing a reduction rather than
trusting a passing corpus.

**(a) The wake test must use the sleeping thread's *own* recorded transition.**
`check_mpor_dependency(active, u)` compares against `thread_last_*[u]`, which
describes the transition `u` took *before* it was put to sleep, not the one it
would take next. Waking off it keeps `u` asleep across genuinely conflicting
transitions. Measured on the first 73 CORE concurrent tests: **19 verdict
changes, every one `FAILED` → `SUCCESSFUL`**. Fixed by storing, per sleeping
thread, the footprint of the transition it took from the node where it was put
to sleep (`execution_statet::transition_footprintt`) — which, for as long as it
stays asleep, is exactly the transition it would take.

**(b) Sleep-marking requires an exhausted subtree.** Adding `t` to `Sleep(s)`
claims everything reachable through `t` was explored. MPOR blocking, a state-hash
collision and the context bound each cut a subtree short, and the claim is then
false. The context bound also breaks the cost symmetry the argument needs: if
`t` is the thread already running at `s`, then `s→u→t` costs one switch and the
supposedly equivalent `s→t→u` costs two, so the covering schedule can fall
outside a budget the skipped one fits in. `reachability_treet::mark_search_truncated`
clears an `exhaustive` flag at each of those three sites and propagates it to the
parent on backtracking. An `interleaving_unviable` break is deliberately *not*
a truncation: that state's guard is false, so everything below it is infeasible.

**(b′) Both halves of the reduction have to be on the same code path.** The
pruning half (`dfs_explore_thread`, `erase_current_frame`) is shared with
`generate_schedule_formula` and `--direct-interleavings`, but the waking half
lives only in `get_next_formula`. On those two paths a thread went to sleep and
never woke. `--sleep-sets` is therefore forced off under `--schedule` and
`--direct-interleavings` rather than half-applied.

**(c) MPOR's read set under-approximated through `printf`.** A global passed as
a `printf` argument was recorded nowhere: the frontend lowers the call to an
OTHER instruction, and only the function-call path ran `analyze_args`. Sleep
sets then called the reader independent of a writer of that same global and
slept through the race (`esbmc-unix/00_race25`). Fixed by overriding
`symex_printf` in `execution_statet`, the same way `symex_goto` / `assume` /
`claim` are already overridden. This one is a pre-existing hole in the
independence relation, not something sleep sets introduced — MPOR happens not to
lose that race, but nothing guaranteed it would not.

**(d) `--data-races-check-only` makes the independence relation vacuous.**
`get_expr_globals` returns before recording anything under that flag, so every
`thread_last_*` set is empty, `check_mpor_dependency` returns false for any pair
of transitions, and a thread put to sleep can never wake — the entry then
propagates to every descendant. Two CORE tests flipped `FAILED` →
`SUCCESSFUL`: `github_2928_qw2004_2` and `github_4423_atomic_race`, the latter
being the reproducer for the race #4423 fixed, lost again. `--sleep-sets` is
forced off there, pinned by `github_6831_sleep_sets_forced_off`. Note the
difference from (c): (c) was a relation missing one access, this is a relation
missing all of them, so no wake test can rescue it.

**(e) A loop the unwind bound truncated looks like an infeasible path.** Under
`--no-unwinding-assertions`, `loop_bound_exceeded` cuts the remaining iterations
with an assumption that drives the state guard false. `get_next_formula`'s
`interleaving_unviable` break then abandons the pending switch — and that break
is deliberately *not* a truncation, because for a genuinely false guard every
schedule below really is infeasible. Here it is not: the iterations are
unexplored, not impossible. `esbmc-unix2/11_podelski.fig3.lics04` went `FAILED`
→ `SUCCESSFUL` (319 → 36 schedules), losing a W/R race on `x`. Fixed by
overriding `note_bounded_loop_truncation` — the hook #6387 already added for
coverage reporting — to clear `exhaustive`. It is a precise fix rather than a
blunt one: the reproducer still gets 67 sleep prunes, and `01_malloc_19` keeps
all of its 255-schedule win. Marking the `interleaving_unviable` break itself
instead would be sound but useless, taking `01_malloc_19` from 3 s to over 300 s.

One non-soundness bug rounds out the list: under `--interactive-ileaves`,
`dfs_explore_thread` can now decline the thread the user chose while
`check_thread_viable` — which explains refusals to that user — knows nothing
about sleep sets, so `decide_ileave_direction` hits its
"selected different thread from user choice" `abort()`. Forced off there too.

**Where it pays.** Because of (b), sleep sets fire only along paths the search
exhausted — so every MPOR block and every context-bound cut costs them ground,
and they pay most with `--no-por` and no `--context-bound`. That is not a
limitation for the target: `esbmc-wrapper.py` passes no `--context-bound`
(§2.2), which is the regime §2.2 shows is not enumerable at all today.

`01_malloc_19` and `01_malloc_21` are the same reproducer family and differ
mainly in that only the latter carries `--context-bound 2`, which makes them a
direct A/B:

| test | bound | base | `--sleep-sets` | `--no-por --sleep-sets` |
|---|---|---|---|---|
| `01_malloc_19` | none | 819 (mpor 661) | 809 | **255**, 6.9 s → 2.6 s |
| `01_malloc_21` | `2` | 919 (mpor 402) | 912 | timeout |

Unbounded, sleep sets alone beat MPOR by **3.2×** with the verdict unchanged;
bounded, they are (soundly) inert. The original exit criterion named
`01_malloc_20` at `--context-bound 2` — the one configuration in which this
reduction cannot fire.

Behind MPOR rather than replacing it the win is smaller but real. Every CORE
test in `esbmc-unix`/`esbmc-unix2` that calls `pthread_create` — 343 of them —
was run twice, once with its own flags and once with `--sleep-sets` added. Of
the 331 that report a schedule count both ways, `--sleep-sets` **cuts 63,
leaves 268 unchanged, raises none, introduces no timeout, and changes no
verdict**. Two more datapoints in the unbounded regime:
`regression/python/threading_thread_increment_race_no_flag_fail` with the bound
dropped **times out** on stock and returns `VERIFICATION FAILED` in 67
schedules with `--sleep-sets`; `esbmc-unix/00_race24` goes 121 → 21.

That sweep is the gate, and it earned its keep twice: an earlier revision of it
over 76 tests caught (a), and the full 343 caught (e) — a single flip that the
76-test corpus did not contain. Sizing the corpus to the whole population of
concurrent tests, rather than to a sample, is what made the difference.

**Exit — restated and discharged for W1.1:** no verdict change across the 343
CORE concurrent tests of `esbmc-unix`/`esbmc-unix2` with `--sleep-sets` on, no
verdict change on the default path either (585 of 586 with stock flags, the one
exception a `FUTURE` test killed by hand), #4584's regression tests still
detecting their race, and a measured reduction in the unbounded configuration.
The remaining candidates (2) and (3) are untouched.

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

**Root cause: the fingerprint omitted `active_thread`.** Dumping every
fingerprint with its components on `race_guard_self_clear` showed three states
sharing one hash with byte-identical value maps *and* identical pcs, differing
only in `active=0` vs `active=1`. Equal pcs and equal values still leave two
states scheduling differently — MPOR's dependency chain and
`decide_ileave_direction`'s scan both key off which thread just ran — and
`post_hash_collision_cleanup()` marks every switch from the survivor explored.

Two earlier hypotheses were tested and **rejected** before this one, and are
recorded so they are not re-tried: the monotone `dynamic_counter` making
post-`malloc` states unique (a controlled probe still pruned 76 states), and
constant propagation hiding values from the map (`assigned_value` is never nil
on `goto_symex_statet::assignment`'s path).

Fixed by mixing `active_thread` into `generate_hash()`
(`execution_state.cpp:1398`) — one line, and finer by construction, so it can
only reduce pruning, never increase it.

**Cost, same corpus re-measured against the fixed binary:**

| | tests where hashing cuts schedules | verdict changes | wall |
|---|---|---|---|
| before fix | 62 of 314 | **4** | −16.5 % |
| after fix | 41 of 314 | **0** | −12.0 % |

The fix gives up some pruning and keeps most of it; the largest win is
untouched (`11_cook.fig2.pldi07` 11,130 → 79 schedules either way). Schedule
counts are deterministic so the 62 → 41 comparison is exact; the wall figures
come from separate runs and carry ~10 % run-to-run noise, so read the sign, not
the 4.5-point gap.

All 287 registered regression tests that pass `--state-hashing` pass with the
fix. `github_6831_hashing_unsound{,_mpor}` are CORE.

**Exit: discharged.** Hashing is sound on this corpus and still pays for
itself, so `--state-hashing` stays in `esbmc_dargs`. Making it prune *harder*
is a separate question, and W1's levers look better-founded than another pass
at the fingerprint.

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

#### W3.1 — The first item is already built: `--smt-during-symex`

`dfs_execution_statet::clone()` (`execution_state.cpp:1631`) deep-copies the
whole target equation at every DFS node — which is why each of the 940 formulas
carries the full trace rather than its suffix (mean 451 SSA assignments, 424,349
across the run, for a program whose single schedule is ~451). Except under
`--smt-during-symex`, where it keeps one shared equation and calls
`push_ctx()` / `pop_ctx()` instead, and `bmc.cpp:2207` stops rebuilding the
solver per interleaving. That *is* "push/pop over the shared prefix".

On `01_malloc_20` with its own flags, three runs each, the schedule count,
counters and verdict identical (940 / 296 MPOR / 0 hash / SUCCESSFUL):

| | baseline | `--smt-during-symex` |
|---|---|---|
| wall | 21.25, 23.62, 21.44 s | 19.20, 19.12, 20.04 s |
| encoding to solver | 1.387 s | **0.398 s** |
| decision procedure | 1.603 s | **1.095 s** |
| symex | 15.408 s | 15.603 s |

~12 %, with the ranges disjoint. The saving lands exactly where the shared
prefix predicts — encoding −71 %, solving −32 % — and symex is untouched, as it
must be when the same 940 schedules are still enumerated.

**So W3's stated exit is already discharged by an existing flag**, which 275
regression tests pass explicitly — and which the SV-COMP wrapper, as W3.3
found after this was written, has been passing all along by implication.
(This paragraph originally read "which the SV-COMP wrapper does not pass";
that was wrong, and W3.3 is the correction.) It also generalises,
which `01_malloc_20` alone could not have shown — the §2.1 mistake. Every CORE
test in `esbmc-unix`/`esbmc-unix2` that calls `pthread_create`, 346 of them, run
twice with its own flags, 120 s cap, 4-way parallel:

- **No verdict changed** on the 329 whose rows survived (194 FAILED, 132
  SUCCESSFUL, 2 no-answer both ways). The single apparent disagreement,
  `pthread_cleanup8_fail`, already passes `--smt-during-symex` in its own
  `test.desc`, so the sweep passed it twice and ESBMC rejected the duplicate.
- **Schedule counts identical on all 323** that report one — as they must be:
  the flag changes how a schedule is encoded, never which are explored. That is
  the cheap check that this is not a reduction in disguise.
- Wall over the 326 answered both ways: 526 s → 455 s, **−13.5 %**, matching
  `01_malloc_20`'s −12 %.

Two methodology caveats: the sweep ran 4-way parallel, so treat −13.5 % as the
ratio rather than a per-test time; and 17 of the 346 rows were torn by
concurrent writes and dropped rather than rerun.

The more important consequence is what it leaves: with encoding and solving
largely removed, symex is **87 %** of BMC time rather than 75 %.

#### W3.2 — Copying is not the lever either (measured, negative result)

The obvious next suspect was the deep copy above: 940 formulas each carrying a
full 451-assignment trace looks like an enormous amount of duplicated state.
Timing both halves of `dfs_execution_statet::clone()` directly (temporary
instrumentation, `01_malloc_20`, 22.78 s run, 1513 clones) says otherwise:

| | total | share of run |
|---|---|---|
| `execution_statet` copy | 0.891 s | 3.9 % |
| target-equation deep copy | 0.190 s | **0.8 %** |

So eliminating the equation copy entirely — which is what `--smt-during-symex`
already does — can only ever be worth 0.8 %, and the 12 % it actually delivers
comes from the solver side, not the copy. Symex's 15.4 s is genuine symbolic
execution of 940 suffixes, not bookkeeping around it.

That closes lever B at the level the plan framed it. Making each schedule
cheaper has roughly 5 % of the run left in it once `--smt-during-symex` is on;
everything else is the schedule count itself, which is lever A. **W1 is
therefore the only remaining lever with headroom**, and W3 should not be
resourced further on the strength of §2.3's 74 % — that share is symex doing
work, not repeating it.

#### W3.3 — The wrapper question was already answered, and the answer crashes

W3.2 left one open decision: does `--smt-during-symex` belong in the SV-COMP
concurrency configuration? **It has been in it all along.** `--smt-symex-guard`
sets `smt-during-symex` (`command_line_options.cpp:442-450`, logging "Enabling
--smt-during-symex to use features that involve encoding SMT during symex"), and
the wrapper's concurrency block passes `--smt-symex-guard`
(`esbmc-wrapper.py:271`). The implication is now pinned by
`github_6831_smt_during_symex_implied` so the wrapper's dependency on it cannot
be refactored away silently.

Re-measured under the wrapper's concurrency flags rather than the ones the tests
carry (W4.2's method note), against a build of `f2df960d08` — 135 commits past
the §2 environment, which is what puts the five `mpor_aggregate_ptr_*` tests
#6981 added on 2026-08-14 inside the corpus at all. All 376 CORE tests across
`esbmc-unix` and `esbmc-unix2` that call `pthread_create` (377 once this
workstream's own test is counted; the sweep predates it), 60 s cap, 8-way
parallel, base = the wrapper's concurrency flags minus
`--smt-symex-guard`, variant = base plus `--smt-during-symex`:

| | result |
|---|---|
| agreed answers | 332 |
| no answer in either arm | 37 |
| wall over the agreed set | 690.0 s → 639.9 s, **−7.2 %** |
| verdict differences | 7, of which **5 are a segfault** |

**The arms isolate the implied flag, not the shipped line.** `--smt-symex-guard`
is not an alias for `--smt-during-symex`: it also queries the solver at undecided
branches (`symex_goto.cpp:34-56`), so the wrapper's actual configuration is a
third arm — and the schedule table below shows it not tracking the variant arm on
two of five tests. The −7.2 % is therefore the value of `--smt-during-symex` in
isolation; the shipped line's own wall number is still unmeasured. Three
corrections to W3.1 follow.

**Schedule counts are not invariant under this flag.** W3.1 reported them
identical on all 323 tests "as they must be: the flag changes how a schedule is
encoded, never which are explored". Five tests contradict that under the
wrapper's flags. Complete schedules (`schedules_explored - pruned_by_mpor -
pruned_by_hash`) summed over the three rounds `--falsify-context-bound 1
--incremental-bmc` runs — all three arms run the same three rounds, so the round
count is not the confound — re-run sequentially under a hard cap:

| test | base | `--smt-during-symex` | `--smt-symex-guard` |
|---|---|---|---|
| `SV_COMP_03` | 8 | **10** | 8 |
| `github_4423_atomic_norace` | 55 | 52 | 52 |
| `github_6478-spinlock` | 32 | 29 | **28** |
| `github_6831_sleep_sets_forced_off` | 50 | 49 | 49 |
| `race_guard_merge_locked` | 65 | **23** | 23 |

So the cheap "not a reduction in disguise" check W3.1 relied on does not hold in
general, and a future measurement must not assume it. Two further readings: the
flag can raise the count as well as lower it (`SV_COMP_03`), and the guard arm
differs from the variant arm on two of the five, which is the direct evidence
that the guard's branch pruning is a separate effect rather than a wrapper for
this one. The mechanism is **not** established — `reachability_tree.cpp:671` is a
place the DFS reads the flag (backtracking clears already-checked assertions and
lowers `remaining_claims` only when it is off), but `schedules_explored` is
incremented independently of claim counts, so that line is a candidate and not a
demonstration. Quoted as a measurement.

**Two of the seven verdict differences were parallel artefacts.**
`02_account_symbolic_02` and `02_phase_06` agree when re-run sequentially under a
hard cap. W4.2's method note firing again; the −7.2 % is likewise a ratio from a
parallel run, not a per-test time.

**The other five are a crash on the shipped command line.**
`mpor_aggregate_ptr_race_symbolic_{offset,offset_locked,skip_mismatch,struct_member}`
and `mpor_aggregate_ptr_widen_contained` — all added by #6981 on 2026-08-14 —
SIGSEGV under `--smt-symex-guard` alone, which is exactly what the wrapper
passes. Bitwuzla and Boolector; Z3 is unaffected, so
`github_6831_smt_during_symex_crash` pins the solver — it is
`mpor_aggregate_ptr_race_symbolic_offset`'s program under the wrapper's flags,
the flags being the only load-bearing difference.

Root cause, from a backtrace taken with an `LD_PRELOAD` SIGSEGV handler (no
debugger on the measurement host): an array of pointers is an array of tuples,
so a symbolic index reaches `array_convt::mk_select`'s case switch
(`array_conv.cpp:174`), whose ite chain projects out of
`tuple_node_smt_ast::elements` (`smt_tuple_node_ast.cpp:195`, reached through
the ite chain at `:86-91`), itself reached from
`dfs_execution_statet::clone()`. `elements` is filled in lazily, so a tuple
created at one context level can hold ASTs allocated at a deeper one, and
`pop_ctx` deletes every AST allocated since the matching push
(`smt_solver.cpp:278-282`). `array_convt::pop_array_ctx` clears its own
select / with / index-map records, but `pop_tuple_ctx` only forwarded to it and
cleared nothing of its own — so a DFS backtrack under `--smt-during-symex` left
those vectors dangling. A scalar array does not reproduce it; the array must
hold tuples.

Two dead ends are worth recording, because both look right. Comparing the level
`elements` was filled at against the current one does **not** work: levels are
reused, and the observed failure is a tuple filled at level 2, popped to 1, then
reached again at level 2. Invalidating on every pop instead is worse than
useless — it is unsound. Elements that are still alive would be rebuilt as fresh
unconstrained symbols, silently disconnecting the tuple from every assertion
already made about it. The fix has to clear exactly the vectors whose contents
the pop destroyed, which is why it is a registry keyed by the level the elements
were *installed* at, and why tuples whose elements belong to their own level are
deliberately left alone: the same pop destroys them too.

**Exit:** ~~measurable wall-time reduction on `01_malloc_20`~~ — discharged by
W3.1 across the concurrent CORE corpus at unchanged verdicts (and, under the
tests' own flags, unchanged schedule counts — W3.3 finds that second half does
not generalise). W3.2 closes the performance question: what remains under lever B
is ~5 % of the run. W3.3 closes the wrapper question — the flag is already in the
concurrency configuration, is worth ~7 % in isolation there, and should stay.
The availability defect W3.3 uncovered is **fixed**: `pop_tuple_ctx` now clears
the element vectors the pop destroyed, all five reproducers answer, and Bitwuzla,
Boolector and Z3 agree on every one. Every registered test whose `test.desc`
passes a flag that implies `--smt-during-symex` — `--smt-symex-guard`,
`--smt-thread-guard`, `--smt-symex-assert`, `--smt-symex-assume` or the flag
itself — passes: 276 such directories exist, 273 are registered with `ctest`, and
all 273 are green. So are all 186 registered tests of `esbmc-unix2` (138 of them
CORE). `github_6831_smt_during_symex_{crash,safe}` pin the FAILED and SUCCESSFUL
verdicts on this path, and both segfault on the pre-fix binary.

#### W3.4 — The sibling hazard is latent, and the invariant behind it is measured

`array_ast::array_fields` looks like the same bug waiting to happen: every site
that writes it does so into an `array_ast` created in the same statement, except
`convert_array_assign` (`array_conv.cpp:40`), which copies into a pre-existing
destination — the exact shape of the tuple `assign` path. It also copies
`base_array_id`, which indexes the containers `pop_array_ctx` resizes, so a
stale destination would be an out-of-range index as well as a dangling pointer.

Both sites rest on one unstated invariant: **an assign destination is
current-level**. It holds because `convert_assign`'s LHS (`smt_solver.cpp:364`)
is an SSA symbol, converted at the level its defining assignment is encoded at
rather than fetched from a shallower cache. Measured rather than argued: a build
instrumented to log every assign whose destination predates the current level
found **zero** across 379 CORE concurrent tests under the wrapper's flags, and
**zero** across all 2133 CORE tests that pass a flag implying a push/pop
strategy, each run with its own `test.desc` flags.

So the hazard is latent, not live, and no machinery is warranted at either site.
What the patch leaves behind is the invariant itself, stated at both sites and
asserted at the tuple one, so a future change to symbol caching trips an
assertion in the Debug CI build rather than a segfault in competition. The
tuple `assign` registration is kept as a defensive fallback and is triaged as
such: it is correct under any input, costs one comparison, and the measurement
above is evidence of unreachability rather than proof of it.

### W4 — Bound the schedule space in the SV-COMP strategy — **investigated, not a flag flip**

§2.2 shows the wrapper's own configuration cannot enumerate this reproducer at
all. `--incremental-context-bound` exists (`options.cpp:553`, "stops at the
first violation or once a round has covered every interleaving") and the
wrapper does not use it.

The proposal was: for the concurrency categories, explore shallow schedules
first and deepen while the budget lasts, so a task that currently times out with
no answer instead answers at the largest bound it can afford. Three findings
change its shape. It is not a wrapper configuration change, and it is not free.

**The soundness constraint is already honoured — this half needs no work.**
`do_context_bound_deepening` (`bmc_strategy.cpp:525`) sets
`suppress-bounded-success` and emits SUCCESSFUL only when `!cs_bound_pruned`,
i.e. only after a round the bound did not truncate; otherwise it reports
VERIFICATION UNKNOWN. A violation at any bound is genuine, each round being an
under-approximation. The oracle is complete: `get_CS_bound()` has exactly one
consumer, `check_if_ileaves_blocked` (`execution_state.cpp:493`), which sets the
flag (`:502`) guarded on a switch actually being available, so a terminal state
does not read as truncated; the flag is per-`reachability_treet` and each round
builds a fresh `bmct`, so it does not leak across rounds. What "covered every
interleaving" does *not* cover is `--unwind` — a SUCCESSFUL is still bounded by
the unwind bound in the ordinary BMC sense, which the log line is careful about
("not bounded by it", the context bound).

**The wrapper cannot adopt the flag: it collides with unwind deepening.**
`--incremental-context-bound` is rejected in combination with `--incremental-bmc`
(`driver.cpp:218`, #6480 — only one driver may own the outer loop), and the
wrapper sends *every* concurrency task through `--incremental-bmc`
(`esbmc-wrapper.py:318`). So W4 is a code change after all, but not for the
reason anticipated above: not a soundness gap, but that the two deepening
loops do not compose. Designing that composition is the remaining W4 work.

**It buys answers on unsafe tasks and loses them on safe ones.** All 40 CORE
unsafe concurrent tests in `esbmc-unix`/`esbmc-unix2`, each run twice with its
own flags minus any `--context-bound`, 120 s cap:

| | direct (unbounded) | `--incremental-context-bound` |
|---|---|---|
| identical, ~1 s both | 37 | 37 |
| `00_rwlock2` | 3 s | 1 s |
| `00_atomicity07` | 13 s | 2 s |
| `00_rwlock4` | **no verdict in 120 s** | **FAILED, 1 s** |

No verdict changed. `00_rwlock4` is the #6480 shape and the SV-COMP shape — a
violation needing few switches, stranded deep in unbounded DFS order;
reconfirmed standalone under `--no-por`, where the direct run produced no
verdict in over 130 s and deepening found it at bound 2 in 0.97 s.

The safe side is the opposite, and the cost is not merely wall-clock. On
`01_malloc_19` (`--no-unwinding-assertions --unwind 3 --force-malloc-success
--memory-leak-check`) deepening converges at bound 10 on the same 1871 schedules
the direct run explores, but re-explores every prefix nine times to get there:
~11,470 schedules cumulative, 86.3 s against 12.3 s — **7×** for an identical
verdict. Under a time cap that turns into lost verdicts rather than slow ones.
Over all 146 CORE safe concurrent tests, 60 s cap, same flag treatment:

| | count |
|---|---|
| SUCCESSFUL both ways | 114 |
| **SUCCESSFUL → TIMEOUT** | **18** |
| no answer either way | 14 |

No verdict *changed* — nothing went SUCCESSFUL → FAILED — so the soundness
argument holds on this corpus; what deepening costs is answers, not correctness.

The threshold is sharper than the count: every one of the 114 kept proofs costs
≤7 s directly, and every one of the 18 lost costs ≥4 s. Read the *ratio* rather
than the 18. That sweep ran 8-way parallel, and re-running two of the losses
sequentially splits them — `github_2174` 6 s → 52 s, which survives a 60 s cap
unloaded, while `00_rwlock1` 15 s → over 90 s is lost at any comparable budget.
So the count moves with the cap and the load; the 7–9× slowdown behind it does
not. Any fixed budget converts the more expensive safe proofs into non-answers.

So the trade is real in both directions and it is not symmetric in value:
deepening bought one stranded falsification in 40 and cost 18 proofs in 132.
A wrapper that simply switched it on would trade correct `true`s for `unknown`s
at roughly ten times the rate it converts a timeout into a `false`. That is an
argument for an adaptive composition — deepen the context bound only where
falsification is the goal, leaving an unbounded exhaustive attempt able to
finish — and against a flag.

**Exit:** unchanged — a measured score delta on the affected categories, plus
the argument for why no `true` is emitted without exhaustive coverage (the
second half is now discharged above). The composition design is the open work.

#### W4.1 — `--falsify-context-bound`, the composition (shipped, off by default)

The wrapper cannot adopt `--incremental-context-bound`: it is rejected
alongside `--incremental-bmc` (`driver.cpp:218`, #6480 — only one driver may own
the outer loop) and the wrapper sends every concurrency task through
`--incremental-bmc` (`esbmc-wrapper.py:318`). The two deepening loops do not
compose because both own the *verdict*, not because they cannot run in sequence.

`--falsify-context-bound N` gives up the half that collides. It deepens the
context bound from 1 to N before the chosen strategy runs, under
`suppress-bounded-success`, so it can only report a violation: each round
under-approximates twice over — the context bound truncates the schedule space,
and the pre-pass forces `--no-unwinding-assertions` (defaulting `--unwind` to 1
when the run sets none) so a truncated loop yields fewer paths rather than a
spurious unwinding-assertion failure. A violation found is therefore genuine
whatever the strategy would have concluded, and finding none is not evidence,
so the strategy afterwards runs untouched.

That argument only holds where a SAT round *means* a violation, which is
narrower than it first appears and cost four wrong verdicts in review before it
was pinned down. `--forward-condition` and `--inductive-step` read SAT as
"unable to prove"; `--termination` inverts it further, since its markers make
reaching an assert evidence that the loop terminates; a `--multi-property`
round owns the property table and would report the truncated pre-pass as the
whole result; and `--partial-loops` removes the very assumption the
under-approximation rests on. The first four are rejected, `--partial-loops` is
forced off for the pre-pass, and coverage runs skip it (its rounds would fold
into the reported figure and print one `[Coverage]` block per bound).

On `00_rwlock4` — the #6480 shape, a violation needing few switches stranded
deep in unbounded DFS order — `--incremental-bmc` alone produced no verdict in
90 s, and `--incremental-bmc --falsify-context-bound 2` reported FAILED in
1.1 s. `--k-induction --falsify-context-bound 2` behaves the same (1.2 s); both
combinations were previously rejected outright.

#### W4.2 — Choosing N, and why the corpus cannot answer it alone

**Measure the configuration you intend to ship, not the one the tests carry.**
`00_rwlock4` is stranded for 90 s only with `--no-por`, which comes from its own
`test.desc`. Under the wrapper's actual concurrency flags — POR left on, plus
`--state-hashing --smt-symex-guard --cswitch-skip-readonly-globals` — the same
benchmark answers in 1.9 s. Every number below is therefore reported twice:
once with each test's own flags, once with the wrapper's.

**Corpus flags, 343 concurrent CORE tests in `esbmc-unix`/`esbmc-unix2`, 60 s
cap.** No verdict changed at N = 1, 2 or 3. Nothing was recovered either — and
the reason is that *this configuration strands nothing*: all 343 answer at base
(200 FAILED, 143 SUCCESSFUL). The sweep bounds the cost and confirms no
regression; it cannot speak to the benefit.

| N | wrong verdict flips | recovered | answers lost | total wall |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 | +4.4 % |
| 2 | 0 | 0 | 1 | +39.3 % |
| 3 | 0 | 0 | 6 | +112.5 % |

The cost concentrates where the pre-pass inherits an expensive configuration:
the worst three (`01_cond_06` +21.7 s, `01_cond_05` +19.4 s, `01_cond_02`
+19.3 s) all pair `--deadlock-check` — which disables the main-thread-ended cut
(`execution_state.cpp:517-521`) and so enlarges the schedule space — with
`--unwind 3`/`4`, and the pre-pass runs its rounds at that same unwind bound.
The wrapper passes no `--unwind`, so its pre-pass runs at 1 and none of this
applies; a user who sets a deep unwind *and* a large N pays for a second full
verification, which is the argument for keeping N shallow rather than for
capping the bound behind the user's back.

**Wrapper flags, same corpus.** No verdict changed at N = 1 or 2 and nothing
was lost here either, but this configuration *does* strand, which is what makes
it the one to decide on. The parallel sweep flagged 26 of 340; re-run
sequentially under a hard cap, **22 are genuinely stranded**, three fail to
parse (their `test.desc` carries `-Wno-error=implicit-function-declaration`,
which substituting the wrapper's flags drops — excluded, not stranded), and one
answers UNKNOWN just past the cap.

Of those 22, **N = 1 recovers exactly one**, and N = 2 recovers nothing further
at higher cost — so **the wrapper takes N = 1**:

- `03_microbenchmark`, under the wrapper's exact command line: no verdict in
  **120 s** at base, **FAILED in 0.71 s** with N = 1. Its own `test.desc`
  expects FAILED at `--context-bound 1`, so the verdict is right as well as
  fast.
- A violation the pre-pass finds still emits a witness — checked separately on
  `00_rwlock4`, 3.7 KB of GraphML — without which the finding would not score.
- Cost on 24 tests that already answer, measured sequentially: median
  **+0.02 s**, p90 +0.25 s, max +0.30 s.

One in 22 is a modest return, and it is quoted as such. What justifies the
change is the shape of the trade rather than its size: the recovered task
converts a timeout into a correct `false` for 0.7 s, no verdict moved in 683
runs across both configurations, and the pre-pass cannot claim a proof by
construction — so the downside is bounded by the median 0.02 s it costs
everything else.

**Method note — `timeout N esbmc` is not a cap.** Several runs in the parallel
sweeps recorded 900+ s against a 60 s limit, and an unrelated ESBMC process on
the same host survived 5 hours under `timeout 20`. ESBMC can outlive SIGTERM,
and `subprocess.run(timeout=)` inherits the same problem through the captured
pipes. Anything timing-sensitive here needs `timeout -s KILL` and a sequential
run; the parallel sweeps are quoted for verdicts only.

**Still open:** the score delta itself. The corpus is a proxy — it is where the
mechanism can be shown to work and shown not to regress, not where SV-COMP
points are won. The exit above is discharged only by a competition run.

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
- **G3 — schedule count pinned. Discharged.** Both `01_malloc_20` at
  `--context-bound 2` (the oracle the bisect used, 940 schedules / 296 MPOR,
  THOROUGH) and a CORE variant `github_6831_schedule_count` at `--unwind 1`
  (262 / 95, 4 s so it runs in PR CI) now assert their counters, so a future
  truncation is a test failure rather than a score movement noticed a release
  later. Re-introducing #6607's search-global `main_thread_ended` collapses them
  to 66 / 15 and 14 / 3 respectively — measured, not assumed, so the pin is
  known to discriminate the exact regression §7 is about.
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
  separate defect, paid by all 36,603 tasks rather than by the concurrency
  categories, and has its own plan:
  [`svcomp-6831-fixed-cost-plan.md`](svcomp-6831-fixed-cost-plan.md).
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

Two workstreams in, the recurring lesson is that every reduction here is
*conditionally* sound and ships without saying on what: state hashing needed the
active thread in its fingerprint (W2), and sleep sets need an exhaustively
explored subtree and an independence relation that misses no access (W1.1).

The second lesson is about how the conditions get found. Four of the six in
W1.1 surfaced as verdict flips on existing CORE tests; the remaining two came
from review of a green tree, and one of those — (e), a truncated loop passing
for an infeasible path — flips a test that a 76-test sample missed and only the
full 343 contained. So neither method dominates: run the sweep over the whole
population rather than a sample, and still review the soundness argument at each
site where a reduction decides a subtree was finished.
