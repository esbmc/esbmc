# Plan — a partial-order encoding for concurrent programs (`--partial-order`)

**Status (2026-11-28):** phases 1 to 3 merged; phase 4 in progress. Phase 0 is
done, with the measurements below (§4), and the maintainer decided **go** on
2026-11-24. `--partial-order` verifies programs on global scalars, and
arrays and structs of them, including through pointers and through global
pointers whose values are globals' addresses, with threads created
in loops, mutexes, condition variables, `pthread_exit` and the scheduler
intrinsics phase 1 lists, without assumptions, and checks data races on
scalars and deadlocks on mutexes and joins. Its counterexamples list steps
in the order the solver's clocks give them, and its violation witnesses carry
thread ids.
A loop or recursion cut by `--no-unwinding-assertions` stops only its own
thread and the threads that wait for it, and so does a lock on a mutex that
is never released. An access inside an atomic section races with a plain
one. A compound assignment to a narrow `_Atomic` object is recorded as a
write, and accesses to an `_Atomic` object never race. A failed
assumption, `abort()` included, stops only its own thread, also from inside
a callee, and a join on a thread that ends returns. A thread's argument
may be a value or a global's address.
Bitwuzla runs `--partial-order` faster than Z3. Clocks are as wide as the
event count needs, and Z3 is faster still with `--ir`, which Bitwuzla cannot
run. Loop guards are not pruned by the interval domain, which would carry one
thread's writes into the next. `--incremental-bmc` and `--falsification`
raise its unwinding bound, and the SV-COMP wrapper runs it on request. The
effort figures are estimates.
**Origin:** [#10296](https://github.com/esbmc/esbmc/issues/10296), which found
`--schedule` unsound and slow, and its
[effort estimate](https://github.com/esbmc/esbmc/issues/10296#issuecomment-6798263610).
[#10847](https://github.com/esbmc/esbmc/pull/10847) and
[#10989](https://github.com/esbmc/esbmc/pull/10989) repaired `--schedule`;
its formula still grows exponentially with `--context-bound`. Tracked in
[#10990](https://github.com/esbmc/esbmc/issues/10990).
**Last updated:** 2026-11-28.

## 1. Goal

A new flag, `--partial-order`, verifies a multi-threaded program with one SMT
formula whose size is polynomial in the number of shared accesses and which
needs no context bound. Symex runs each thread once, without context switches.
Each shared access becomes an *event*, and constraints added after symex let
the solver choose the interleaving: a clock per event, a reads-from choice per
read, and a total order on the writes to each object. This is the approach of
CBMC's `partial_order_concurrency.cpp` and `memory_model_sc.cpp`, under
sequential consistency.

`--schedule` is left as it is. Whether it is removed is decided in phase 4
(§4), once there are measurements.

## 2. Baseline

From #10296, on a two-thread mutex counter (each thread runs
`lock; x = x + 1; unlock`):

| `--context-bound` | default | `--schedule` |
|---|---|---|
| 2 | 3 s | 6 s |
| 3 | 7 s | 29 s |
| 4 | 10 s | >150 s |

Phase 0 is judged against the default column, not against `--schedule`.

## 3. Where it attaches

- **Thread execution.** `reachability_treet` and `execution_statet`
  (`src/goto-symex/scheduler/`) already place every thread's steps in one
  equation under `--schedule`. Under `--partial-order` each thread runs to
  completion in a fixed order (`main`, then each spawned thread), with no
  clone and no backtracking. `goto_symext::intrinsic_spawn_thread`
  (`engine/builtin_functions/threads.cpp`) queues the start function instead
  of offering a context switch.
- **Shared-access detection.** `execution_statet::analyze_assign`,
  `analyze_read` and `get_expr_globals` (`scheduler/execution_state.cpp`)
  already find the shared objects an access touches, including those reached
  through a pointer (#9538). They emit events here instead of context-switch
  points.
- **SSA renaming.** A shared read normally renames to the current thread's
  latest L2 version (`level2t::rename`, `state/renaming.h`). Under
  `--partial-order` it gets a fresh, unconstrained L2 version; a shared write
  gets its usual new version. Both are recorded as events.
- **Event steps.** `symex_targett::step_typet` (`equation/symex_target.h`)
  gains `SHARED_READ`, `SHARED_WRITE`, `SPAWN`, `JOIN`, `ATOMIC_BEGIN` and
  `ATOMIC_END`. `sourcet::thread_nr` already tags every step with its thread.
- **Memory-model pass.** A new `equation/memory_model_sc.{h,cpp}` runs on the
  finished equation, before slicing and solving, and appends the order
  constraints as ASSUME steps, following the existing `ssa_step_algorithm`
  passes.

## 4. Phases

### Phase 0 — prototype (2–3 weeks, go/no-go)

Scope: global scalars only; no pointers, arrays, data-race check or witnesses.
`--partial-order` rejects anything outside this with an error.

- Events for reads and writes, `pthread_create`, `pthread_join`, and the
  `__ESBMC_atomic_begin`/`__ESBMC_atomic_end` pair the mutex model uses
  (`src/c2goto/library/pthread_lib.c`).
- Integer clocks, a reads-from choice per read, a total write order per
  object, and contiguous atomic sections.
- Compare with the default on the §2 program and on about 50 SV-COMP
  ConcurrencySafety tasks restricted to global scalars.

Done: `--partial-order` and its flag conflicts; a scope check on user code
(`goto-programs/partial_order_scope.cpp`) that names the first unsupported
construct; the five event step types, recorded by `symex_targett::event`.
Symex runs each thread to completion (`generate_partial_order_formula`) and
records user-global reads, under a fresh L2 name, and writes as events;
`--program-only` shows them. The operational model's state is still renamed
sequentially, so `pthread_join` waits on the spawn/join events of a later unit.
`equation/memory_model_sc.cpp` prepends hidden assumptions: a clock per event,
program order, a write order per object, and a reads-from choice per read with
its from-read constraints. A spawned thread's events follow its SPAWN event
and precede every JOIN on it; symex replaces `pthread_join` by a JOIN event,
so joins need a thread id that is not a global. No other thread's event falls
inside a user atomic section. Verification still rejects locks and
assumptions.

The three `github_10296_*` programs become `partial_order_github_10296_*`
pairs. Locks are still rejected, so the counter uses an atomic section.

Measurements (Z3, one core each, 2026-11-24). The §2 counter with an atomic
section in place of the mutex, N threads:

| N | default | `--context-bound 4` | `--partial-order` |
|---|---|---|---|
| 2 | 0.6 s | 0.6 s | 2.4 s |
| 3 | 10.7 s | 12.1 s | 10.1 s |
| 4 | >300 s | 269 s | 19.9 s |
| 5 | >300 s | >300 s | 58.4 s |

With the mutex, the default takes 1.1, 2.9 and 3.5 s at `--context-bound` 2,
3 and 4. Of 367 SV-COMP `unreach-call` tasks in `pthread*`, the scope check
rejects 113 and `--partial-order` decides 58 within 30 s, all from
`pthread-wmm`. On those 58, at 60 s each, it gets every verdict right with a
median of 14.1 s; the default decides 54 (median 1.3 s) and
`--context-bound` 2, 3 and 4 decide 58, 58 and 55 (medians 0.3, 0.5 and
0.5 s), with no wrong verdict.

Decision: **go**, by the maintainer on 2026-11-24.

**Go** if every verdict matches and the time grows more slowly than the
default's as the bound rises. **No-go** closes #10990 with the measurements
recorded here.

### Phase 1 — real programs (4–6 weeks)

- Arrays, and pointer dereferences expanded to one event per object in the
  value set.
- Threads created in a loop: one instance per unwinding.
- `pthread_mutex_*`, condition variables, `pthread_exit`.
- The scheduler-dependent intrinsics: `__ESBMC_yield` becomes a no-op;
  `__ESBMC_switch_to`, monitors and LTL are rejected;
  `__ESBMC_blocked_threads_count`, `__ESBMC_terminate_thread` and
  `__ESBMC_really_atomic_*` are ported or rejected one by one.

Done: a global array of scalars is one object. A read of an element reads the
whole array; an element write is an update, a write that also reads the
array's previous value, and the from-read constraint keeps every other write
out from between the two. A dereference reads, or updates, every global its
pointer's value set reaches; a global pointer and a global's address passed to
a function without a body are still rejected. A thread created in a loop is
one thread per unwinding, spawned under that unwinding's guard; a join whose
thread id differs between paths waits, on each path, for the thread it names.
`pthread_mutex_lock` and `pthread_mutex_unlock` on a global mutex are LOCK
and UNLOCK events; a lock is taken only if every earlier lock of its mutex was
released before it. An execution in which a thread waits forever for a lock is
not modelled; phase 2's deadlock check covers it. `pthread_cond_wait` is an
UNLOCK and a LOCK of its mutex: POSIX allows spurious wakeups and the
operational model takes one at will, so signals and broadcasts change nothing.
`pthread_exit` ends the path of the thread that calls it, main included.
`__ESBMC_terminate_thread` in user code does the same, and
`__ESBMC_really_atomic_*` opens and closes an atomic section like
`__ESBMC_atomic_*`; a read of the operational model's thread counters, such as
`__ESBMC_blocked_threads_count`, is rejected.

### Phase 2 — properties (3–4 weeks)

- Data race: two conflicting events from different threads, at least one a
  write, that the clocks do not order.
- Deadlock: no order lets the program finish.
- Rejected at first: atomicity, lock-order, the end-of-program leak check,
  k-induction and the incremental modes.

Done: `--data-races-check` claims, for each pair of conflicting events from
different threads, that no execution runs them back to back. The SC
constraints already order every such pair, so "unordered clocks" never
holds. Global arrays are rejected under the check, since an array is one
object.

`--deadlock-check` frees a stop time: the events before it are a prefix, and
only its locks need a release. The claim is that no prefix leaves main, and
every other thread with events left, waiting on a held mutex or on a join;
the other claims need every thread to finish. Condition waits, which wake at
will here, and `pthread_exit`, which lets threads outlive main, are rejected
under the check.

### Phase 3 — traces and witnesses (1–2 weeks)

- `build_goto_trace` orders steps by their clock values in the model.
- Violation witnesses carry thread ids, reusing the waypoint threading from
  #10766.

Done: each event keeps its clock in its step, and `build_goto_trace` merges
the threads' steps by clock. A step takes the clock of its thread's next
event; a thread with no event follows its spawn. `--witness-output*` is
accepted: each spawn moves to its thread's first step when the merge put that
step first, so the YAML waypoints carry thread ids in clock order.

### Phase 4 — validation and tuning (3–4 weeks)

- Verdict comparison with the default on `regression/esbmc-unix` and
  `regression/esbmc-unix2`.
- A benchexec run on SV-COMP ConcurrencySafety (`needs-svcomp-run`).
- Integer vs bitvector clocks; Z3 vs Bitwuzla.
- Decide whether the SV-COMP wrapper uses `--partial-order` in any category
  and whether `--schedule` is removed.

Done: `scripts/dev/partial_order_compare.py` runs each pthread test of a suite
with and without `--partial-order` and lists the disagreements. On
`esbmc-unix` and `esbmc-unix2` (564 tests, Z3, 20 s each, 2026-11-25), the
scope check or a flag conflict rejects 374, both modes decide 173 and agree
on 162 (median 0.29 s default, 0.26 s `--partial-order`), and
`--partial-order` times out on 9 that the default decides and runs out of
memory on `03_microbenchmark`. The 11
disagreements:

- Missed violations. `esbmc-unix2/01_pthread46` (`--unwind 2
  --no-unwinding-assertions`): two threads spawned in `main`'s `while(1)`
  loop both run `s++; assert(s == 1)`; the unwinding assumption that ends
  `main` also hid the threads' failure. Fixed: past its bound a thread
  records a HALT event instead of that assumption, and a join on a halted
  thread does not return, so only the joiner's later events and claims
  drop out; a thread halted inside an atomic section keeps every other
  event before it (`partial_order_loop_bound*`). `--deadlock-check` keeps
  the assumption. A lock may also wait forever: the thread halts before it,
  so a mutex that is never released, because its holder stopped at a bound
  or ended, no longer hides the other threads' earlier violations
  (`partial_order_lock_*`). A recursion past its bound now also records a
  HALT and ends its thread's path (`partial_order_recursion_bound*`).
  `github_2386_atomic_write_race_fail`: under `--data-races-check`, no race
  was reported between a write inside an atomic section and a plain read in
  `main`. Fixed: a section's bounds no longer separate a race, and two
  accesses inside atomic sections never race (`partial_order_atomic_race*`).
- False alarms. `github_4434_thread_local`: a `__thread` global was shared.
  Fixed: it is no longer an event, so each thread keeps its own SSA chain
  (`partial_order_thread_local*`).
  `github_9441_atomic_narrow_rmw`: a compound assignment to an `_Atomic`
  object narrower than int assigns to `(int)c`, which was not recorded as a
  write. Fixed: writes, element updates and dereference updates see through
  the cast (`partial_order_atomic_rmw*`).
  `esbmc-unix2/11_time_var_mutex`: a lock built from `abort()` in an atomic
  section did not keep a later thread out, since an assumption only
  constrained the claims after it in the equation. Fixed: a failed
  assumption halts its thread, as a loop bound does, and assumptions are
  accepted except under `--deadlock-check` (`partial_order_abort*`,
  `partial_order_assume_*`); the task now takes 100 s.
- Different semantics. In the four `github_6831_{stats,sleep_sets}*` tests
  the default runs `x++` as one step, so a lost update is not reachable;
  `--partial-order` reads and writes separately and finds it.
  `github_6474_reinit` re-initialises a locked mutex: the default unlocks
  it, `--partial-order` keeps it locked and reports a deadlock.
  `02_account_symbolic_05` runs the default at `--context-bound 1`.

Z3 against Bitwuzla (2026-11-27, one build with both, `--extra-args` of the
compare script). On the §2 counters, `--partial-order` takes, for 2 to 5
threads, 2.7, 5.1, 17.4 and 42.7 s with Z3 and 0.8, 3.4, 5.6 and 9.9 s with
Bitwuzla using atomic sections, and 1.0, 5.8, 58.2 and >120 s with Z3 and
0.5, 1.8, 5.0 and 22.1 s with Bitwuzla using the mutex. The default with
either solver passes 120 s from 4 threads with atomic sections and from 3
with the mutex. Variants without the last join are FAILED in every mode, in
under 0.5 s. On `esbmc-unix` and `esbmc-unix2` at 20 s, `--partial-order`
decides 175 tests with Z3 and 182 with Bitwuzla; Z3 times out on 7 that the
default decides, Bitwuzla on 1 (`03_arithmetic_progression_03`). Both
solvers give the same 11 disagreements: the 7 above that remained before
the `11_time_var_mutex` fix, and 4 false data races on an `atomic_uint`
(`github_2149_pthread_{2,6}`, `github_3955_{2,5}`), which C11 5.1.2.4p25
excludes. Fixed: the race check skips an object whose type is `_Atomic`
(`partial_order_atomic_object_race*`).

Clocks were 32-bit bitvectors. They now have the fewest bits that hold one
value per event and a stop time above them all. With Z3 (one core,
2026-11-27), the §2 counter with an atomic section takes 0.2, 0.2, 0.5 and
3.7 s for 2 to 5 threads, against 3.3, 5.3, 17.3 and 47.2 s at 32 bits. The
mutex counter is unchanged (55 s against 56 s at 4 threads), and so are the
`esbmc-unix{,2}` comparison's verdicts and times (75.8 s against 77.8 s in
total on the 175 tests decided). The Bitwuzla times above are at 32 bits;
narrow clocks under Bitwuzla are measured below.

Rerun with narrow clocks (Z3 4.13.3, 20 s each, 2026-11-27), the comparison
decides 211 tests in both modes and finds two more disagreements.
`esbmc-unix2/11_dekker` was a false alarm: symex runs the threads in turn,
and the online interval domain that prunes loop guards carried thread 1's
last write, `flag1 = 0`, into thread 2's loop guard, so thread 2 skipped its
wait loop. Fixed: that pruning is off under `--partial-order`, and
`--interval-symex-assert`, which prunes claims with the same domain, is
rejected (`partial_order_loop_guard*`). `esbmc-unix2/11_bluetooth` runs the
default at `--context-bound 2`; without the bound, or at 4, the default is
FAILED too.

Integer clocks. `--ir` makes every bitvector an integer, the program's own
data included, so it measures integer clocks only together with integer
data. On the §2 counters (thread ids in an array, Z3, 120 s), for 2, 3, 4,
5, 6 and 8 threads, `--partial-order --ir` takes 0.3, 0.2, 0.2, 0.3, 0.8 and
68.2 s with atomic sections, against 0.3, 0.3, 0.4, 4.5, 19.6 and >120 s
with bitvector clocks, and 0.3, 0.2, 0.3, 0.3, 0.5 and 31.3 s with the mutex,
against 0.3, 0.5, 0.6, 17.4, 76.2 and >120 s. Variants without the last join
are FAILED in under 0.5 s with either encoding. `--ir` does not change the
default's times, which pass 120 s from 4 threads with atomic sections and
from 3 with the mutex. On `esbmc-unix{,2}` both encodings give the same
verdicts; on the 212 tests `--partial-order` decides with both, it takes
58.7 s in total with `--ir` against 72.7 s with bitvectors.
`03_microbenchmark` runs out of memory with `--ir` instead of timing out.

Narrow clocks under Bitwuzla (Bitwuzla 0.9.0 and Z3 4.8.12 in one build, two
runs at a time on four cores, 2026-11-28). On the same counters, for 2, 3, 4,
5, 6 and 8 threads, `--partial-order --bitwuzla` takes 0.3, 0.3, 0.7, 1.9,
5.1 and >120 s with atomic sections and 0.3, 0.4, 0.5, 2.6, 8.1 and >120 s
with the mutex. Z3 on the same build takes 0.2, 0.3, 0.5, 4.5, 21.2 and
>120 s, and 0.3, 0.3, 0.7, 15.1, 67.1 and >120 s; with `--ir`, 0.2, 0.3,
0.3, 0.3, 0.8 and 62.2 s, and 0.2, 0.3, 0.3, 0.3, 0.7 and 35.9 s. The default
with Bitwuzla takes 0.4 and 6.9 s for 2 and 3 threads with atomic sections
and 2.3 s for 2 with the mutex, and passes 120 s beyond. Every variant
without the last join is FAILED in under 1.5 s. So narrow clocks help
Bitwuzla as they helped Z3, and Bitwuzla stays ahead of Z3 on bitvectors
from 5 threads; Z3 with `--ir` is fastest of all, and `--ir` rejects
`--bitwuzla`. On `esbmc-unix{,2}` (20 s each) with Bitwuzla,
`--partial-order` decides 219 tests and both modes decide 218, which take
71.7 s in total under `--partial-order` against 135.3 s under the default.
The 7 disagreements are the 6 semantic differences above and
`11_bluetooth`. `--partial-order` times out on `03_wronglock_01`, which the
default decides in 1.7 s, still runs out of memory on `03_microbenchmark`,
and decides `02_phase_02` in 0.3 s, where the default times out.

`--partial-order` runs under `--incremental-bmc` and `--falsification`: each
bound is one `--partial-order` run, and the forward condition proves the
program once no thread's loop passes the bound (`partial_order_incremental_bmc*`,
`partial_order_falsification_fail`). These strategies do not propagate
arrays, so a join on `id[j]` reads the ids' array in its event's guard; the
slicer now counts that read, where it had dropped the stores to the array
(`partial_order_incremental_join*`). `--k-induction` is still rejected.

The SV-COMP wrapper's `--partial-order` flag runs a threaded `unreach-call`
or `no-data-race` task as `--partial-order --incremental-bmc`, and falls back
to the default command when ESBMC rejects the task; `--sv-comp` no longer adds
falsification rounds under `--partial-order` (`partial_order_sv_comp*`). Two
ConcurrencySafety run definitions pass it, for the benchexec run.

That run's first attempt gave 689 wrong `false` verdicts, all from
`--smt-symex-guard`, which `--sv-comp` turns on. It and the other
`--smt-symex-*` flags ask the solver during symex, where threads run one after
another, so the answers hold for that order only and pruned feasible
interleavings. `--partial-order` now turns all four off
(`partial_order_smt_symex*`, `partial_order_sv_comp_guard*`).

A halt hid violations. Once any thread halted, a join on a thread that never
halts was taken never to return, since the disjunction of no halts is true,
and the joining thread's later claims held. So main's `reach_error()`, which
`--enable-unreachability-intrinsic` turns into a claim and `assume(0)`, was
never reported. Fixed: such a thread ends (`partial_order_halt_*`).

SV-COMP subset. `partial_order_compare.py` also runs SV-COMP task
definitions, with the phase 0 options. On the 405 `unreach-call` tasks of
the 12 `c/pthread*` directories phase 0 used (Z3 4.13.3, 30 s, 4 jobs,
2026-11-28), `--partial-order` gave 37 wrong verdicts, all false alarms and
36 of them in `pthread-wmm`. `assume_abort_if_not` calls `abort()` in a
callee, and `pop_frame` restored the caller's entry guard, which dropped
the halt from it; a loop bound in a callee did the same. Fixed: a frame
records the halt and returning keeps the merged guard
(`partial_order_halt_callee*`). Now `--partial-order` decides 225 tasks
(210 from `pthread-wmm`, median 8.6 s) and the default 199 (median 3.2 s),
every verdict right; both decide 126 and agree on all of them. The scope
check rejects 101, 91 for a global that is not a scalar or an array of
scalars; 64 time out and 15 run out of memory, against 162 and 44 for the
default.

A global struct whose fields are scalars, arrays or structs of them is one
object, as an array is: a field read reads the struct, and a field write,
directly or through a pointer, updates it. `pthread_` types stay out, since
LOCK and UNLOCK events order them, and `--data-races-check` rejects a global
struct as it does an array (`partial_order_struct*`). On the 473
`unreach-call` tasks of the 26 `c/pthread*` directories of sv-benchmarks at
`67f8e5a3`, this moved no task into scope: the scope check rejects 136, 86
of them for a non-scalar global, and 58 of those, all in `pthread-wmm`, for
an `int *` global such as `x$read_delayed_var`.

A global pointer is now an event object too, if every value assigned to it,
its initialiser included, is a global's address or null. A read may see any
thread's write, so its value set becomes all of those values; taking the
pointer's address is rejected (`partial_order_global_pointer*`,
`partial_order_reject_pointer_*`). The 58 tasks enter scope and the scope
check rejects 78. On them (Z3 4.8.12, 30 s, 4 jobs, 2026-11-28; all expect
FAILED) `--partial-order` decides 29 (median 16.1 s on those both decide),
times out on 28 and runs out of memory on `safe008_power.opt`; the default
decides 45 (median 0.3 s). Both decide 26 and agree on all of them.

`pthread_create` passes a thread argument other than NULL if it is a value,
which the thread receives as it was at the spawn, or a global's address, which
the thread dereferences as any pointer to a global. A pointer that may reach
a local is still rejected, since the thread would read the local's last value
in its owner's run (`partial_order_thread_arg*`). On the 405 SV-COMP tasks,
no first rejection changes: the 30 tasks whose sources pass an argument other
than NULL, 10 of them an address, are rejected earlier, 25 for a global that
is not a scalar or an array of scalars.

## 5. Tests

Every construct a phase adds gets a `VERIFICATION FAILED` /
`VERIFICATION SUCCESSFUL` pair under `regression/esbmc-unix/partial_order_*`,
and the same program run without `--partial-order` must give the same
verdict. The three `github_10296_*` programs are the first pairs. A Catch2 test
under `unit/goto-symex/` checks the constraints the memory-model pass emits for
a hand-built two-thread equation.

## 6. Risks

- **Pointers and arrays.** A shared dereference expands to one event per
  object the pointer may reach, and a whole-array write multiplies them.
  Value-set precision decides whether phase 1 is affordable; CBMC has the same
  weakness.
- **Loops.** Loops in threads grow the event count, and the reads-from choices
  grow quadratically with it.
- **Operational model.** The pthread model reads scheduler state such as
  `__ESBMC_num_threads_running`. Parts of it may need a `--partial-order`
  variant rather than a port (`docs/agents/cpp-standard.md` applies).
- **Symex guards.** The Θ(N²) symex guard cost still applies, now to one larger
  equation.

## 7. What this plan does not establish

- That the encoding beats the default exploration on any input. Phase 0 exists
  to measure that, and may end the plan.
- Weak memory models. TSO and PSO would add 2–3 weeks; the SV-COMP
  `pthread-wmm` tasks are already instrumented and do not need them.
- Effort figures beyond estimates; none has been checked against a prototype.
