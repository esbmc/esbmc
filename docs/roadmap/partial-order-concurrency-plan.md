# Plan — a partial-order encoding for concurrent programs (`--partial-order`)

**Status (2026-11-24):** phase 0 in progress. The `--partial-order` flag, its
scope check, the event step types, the run-to-completion symex path and the SC
memory-model pass exist; it verifies programs without joins, locks, atomic
sections or assumptions. Nothing has been measured; the effort figures are
estimates.
**Origin:** [#10296](https://github.com/esbmc/esbmc/issues/10296), which found
`--schedule` unsound and slow, and its
[effort estimate](https://github.com/esbmc/esbmc/issues/10296#issuecomment-6798263610).
[#10847](https://github.com/esbmc/esbmc/pull/10847) and
[#10989](https://github.com/esbmc/esbmc/pull/10989) repaired `--schedule`;
its formula still grows exponentially with `--context-bound`. Tracked in
[#10990](https://github.com/esbmc/esbmc/issues/10990).
**Last updated:** 2026-11-24.

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
  gains `SHARED_READ`, `SHARED_WRITE`, `SPAWN`, `ATOMIC_BEGIN` and
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
its from-read constraints. Until spawn events order a thread after its
creation, a thread may also be placed before main's initialisation, which can
only add spurious failures; verification rejects joins, locks, atomic sections
and assumptions, which need events the pass does not have yet.

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

### Phase 2 — properties (3–4 weeks)

- Data race: two conflicting events from different threads, at least one a
  write, that the clocks do not order.
- Deadlock: no order lets the program finish.
- Rejected at first: atomicity, lock-order, the end-of-program leak check,
  k-induction and the incremental modes.

### Phase 3 — traces and witnesses (1–2 weeks)

- `build_goto_trace` orders steps by their clock values in the model.
- Violation witnesses carry thread ids, reusing the waypoint threading from
  #10766.

### Phase 4 — validation and tuning (3–4 weeks)

- Verdict comparison with the default on `regression/esbmc-unix` and
  `regression/esbmc-unix2`.
- A benchexec run on SV-COMP ConcurrencySafety (`needs-svcomp-run`).
- Integer vs bitvector clocks; Z3 vs Bitwuzla.
- Decide whether the SV-COMP wrapper uses `--partial-order` in any category
  and whether `--schedule` is removed.

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
