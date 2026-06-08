# Plan: C++ exception lowering — finish the work the engine removal started

Status: **in progress — the imperative engine is already deleted.** The GOTO
lowering is now the sole exception path. What remains is to close the capability
and soundness gaps the deletion exposed (see §8 for the live work). It folds in
an adversarial design review (the "Soundness constraints" and "Scope flags"
sections are the load-bearing parts — read them before implementing).

This document originally described how to make ESBMC's GOTO-level C++ exception
**lowering** reach parity with the imperative engine so the latter could be
deleted. The deletion happened ahead of that parity: the lowering became
unconditional and `symex_catch.cpp` (with `stack_catch`, `symex_throw*`, the
`THROW`/`CATCH` symex cases, and the `--lower-exceptions` flag) was removed. So
the framing has flipped — the items §7 once called "gating before deletion" are
now **hard rejects in production**: a program the lowering declines is reported
unsupported and rejected by symex, not silently handled by a fallback. The
remaining work is therefore about *correctness and coverage of the lowering*,
not about enabling a deletion that has already occurred.

Related: issue #5075 / #5108 / PR #5127 (lowering foundation + `noexcept`/`throw()`
lowering) and the `THROW_DECL`→metadata + frame-boundary enforcement work
(PR #5126), which this plan *consumes*.

---

## 1. Goal and target architecture

There is now **one** implementation of C++ exception handling:

- **GOTO lowering + OM** — `src/goto-programs/remove_exceptions.cpp` rewrites
  throw/catch into guarded control flow over per-thread global exception state
  (`exception_globals.h`: `$esbmc_exc_{thrown,typeid,value}` and the
  `$esbmc_exc_uncaught_count`). The terminate/unexpected/uncaught surface is a
  **compiled OM** in `src/c2goto/library/exception.cpp`
  (`std::terminate`/`set_terminate`, `std::unexpected`/`set_unexpected` via
  `__ESBMC_run_unexpected`, `std::uncaught_exception(s)`). The earlier
  duplicate-thunk problem that forced a header-only model did not recur with the
  current narrow OM. `<exception>` (`src/cpp/library/exception`) still holds the
  standard surface declarations plus the `nested_exception` / `throw_with_nested`
  / `rethrow_if_nested` helpers; of its executable bodies, `current_exception`
  and `rethrow_exception` are still unimplemented placeholders.

The deleted imperative engine was `src/goto-symex/symex_catch.cpp`
(`stack_catch`, `symex_throw*`, `enforce_exception_specifications`,
`symex_throw_bad_cast`, bespoke `terminate_handler()`/`unexpected_handler()`
lookup) plus the `THROW`/`CATCH` dispatch cases in `symex_main.cpp`.

**Target (now largely realised):** no exception-specific dispatch logic in
symex. Exception-library functions execute as ordinary modeled code; the
lowering owns control flow and exception-state bookkeeping. The only thing
"below the OM" is a lowering-visible *raise* primitive (still to be unified —
§8 step 1), not the old engine.

One architectural rule is non-negotiable:

- **Frontend** emits explicit exception operations (real `THROW`, `CATCH`, or a
  dedicated lowering-recognized raise intrinsic), never a hidden helper whose
  semantics only symex understands.
- **Lowering** owns propagation, handler search, and function-boundary policy.
- **OM** owns the standard `<exception>` entry points and handler storage.

Internal helpers use ESBMC-private names (`__ESBMC_*`), not `__gnu_cxx` /
`__verbose_terminate_handler` compatibility names. Note `__ESBMC_`-prefixed OM
helpers that carry a real body (e.g. `__ESBMC_run_unexpected`) now execute as
ordinary calls — symex no longer aborts on a prefixed call when the symbol has a
body (`run_intrinsic` fallback, `symex_main.cpp`).

Two-layer division of labour:

- **Lowering layer**: control flow; updates the runtime exception state
  (including the uncaught count); enforces `noexcept` / dynamic specs;
  materializes thrown objects.
- **OM layer**: handler registration; `terminate`; `unexpected`;
  `uncaught_exception(s)` (done); `exception_ptr`; `current_exception`;
  `rethrow_exception` (still placeholders).

Remaining OM-split follow-ups:

1. **Implement `exception_ptr`** (`current_exception` / `rethrow_exception`),
   today header-only placeholders (§8 steps 1–2).
2. **Keep only standard surface in `<exception>`**; private helper storage stays
   ESBMC-internal and does not leak libstdc++/libc++ naming.

---

## 2. What already exists (reuse, don't reinvent)

- `exception_globals` (lowering): `$esbmc_exc_thrown` (bool, = "active exists"),
  `$esbmc_exc_typeid` (size_t), `$esbmc_exc_value` (void*).
- `wire_throw` already copies the thrown object into a **stable static slot** and
  points `$esbmc_exc_value` at it, so the object outlives the throwing frame.
- A bare `throw;` already re-raises from those globals (clear-on-catch resets
  only the `thrown` flag).
- The OM already stores/dispatches `current_terminate_handler` /
  `current_unexpected_handler`.
- `exception_typeidt` registry: reflexive-transitive subtype closure from `bases`
  metadata + THROW `exception_list` chains.
- PR #5126 metadata: `exception_specificationt` on the function type
  (`kind` ∈ {potentially_throwing, non_throwing, dynamic}, `allowed_types`),
  cached on `goto_functiont` and the symex frame.

The genuinely **new** runtime state is therefore only: the handled-exception
stack, the propagation count, and the `exception_ptr` slot table.

Reuse the `$esbmc_exc_*` names; do not introduce parallel `active_*` globals.

---

## 3. Runtime exception state (single source of truth)

Lowering writes it; OM reads it.

### 3.1 Per-thread state
- **Active (propagating) exception**: `{type_id, object, exists}` — already the
  `$esbmc_exc_*` trio.
- **Handled-exception stack**: stack of `{type_id, object}` for exceptions
  currently being handled (inside a `catch`). Needed by `current_exception()`
  and `throw;`.
- **`active_propagation_count`** (uncaught count): backs `uncaught_exception(s)`.

C++ defines propagation and `uncaught_exceptions` per *current thread*
([except.uncaught]); all three MUST be per-thread. ESBMC plain globals are
*shared* across threads, so these need thread-local storage (or per-thread-indexed
arrays). A global handled stack is **unsound** under interleaving (thread B's
catch entry would pop thread A's frame) — which is why the existing
`$esbmc_exc_*` globals (thrown/typeid/value/uncaught count) are already marked
thread-local.

### 3.2 Shared (cross-thread) state
- **`exception_ptr` slot table**: slots hold `{type_id, object, valid}`.
  `exception_ptr` is a small **integer handle** (index into the table), not a raw
  pointer (avoids solver aliasing reasoning).
- The table is **global** because an `exception_ptr` can be captured in one
  thread and rethrown in another (`std::async`, promise/future).

### 3.3 Slot-table invariants (soundness-critical — see §5)
- **Permanent identity**: a slot index, once handed out, refers to the same
  exception for the entire bounded run. **Never reuse** slot indices.
- **Atomic publish**: the multi-field write that makes a slot observable
  (`type_id`, `object`, `valid`) must be observationally atomic w.r.t. other
  threads, or a torn read makes `rethrow_exception` raise a half-built / wrong
  exception.
- **No reclamation in v1**: never-free. Sound under never-reuse; the only thing
  lost is exception-object destruction *timing* (see §5.3).

---

## 4. OM function semantics (execute as ordinary code)

### 4.1 `std::terminate()` — **implemented** (`c2goto/library/exception.cpp`)
- Loads the installed terminate handler (honouring `std::set_terminate`), calls
  it, and asserts on return/throw; the default handler asserts "terminate called
  after throwing an exception" then `__ESBMC_assume(0)`. Never returns.
- Outcome classification (§5.4) is settled: `noexcept`-escape is a property
  violation; an `abort()`/`exit()` in a handler is silent valid termination.
- Fallback: when the OM `std::terminate` is not linked (a program that does not
  reference the exception library), `emit_terminate` lowers the terminate point
  to a direct assertion instead of an OM call, so the check is never silently
  dropped.

### 4.2 `std::unexpected()` (C++14 legacy only)
- Just run `current_unexpected_handler` if installed.
- The **library cannot** decide dynamic-spec recovery — it depends on the
  violated function boundary. So the **lowered call site** decides:
  - wrap the handler call in a synthesized catch-all;
  - handler returns → `terminate()`;
  - handler throws a type allowed by the violated function's spec
    (`exception_specificationt::allowed_types`, from PR #5126) → continue
    propagation with the new exception;
  - else if `std::bad_exception` is allowed → substitute it;
  - else → `terminate()`.
- Dynamic specs are **C++14-only** (rejected under C++17), so this branch is
  narrow. The lowering now enforces `throw(T...)` and drives `unexpected`
  directly (`build_dynamic_spec_check`); there is no longer a symex path for it.

Status: **implemented.** The OM `std::unexpected` runs the installed handler via
`__ESBMC_run_unexpected`; the lowered call site (`build_dynamic_spec_check`) does
the replacement-throw check and `bad_exception` substitution, and decrements the
uncaught count by one when a replacement throw takes over (the replacement
replaces the original, so exactly one exception stays uncaught).

### 4.3 `current_exception()`
- Active propagating exception present → handle to it.
- Else inside a `catch` → handle to top of the handled stack.
- Else → empty `exception_ptr`.
- Must be readable **inside** a terminate/unexpected handler and return the
  provoking exception (§5.5).

### 4.4 `rethrow_exception(ep)`
- Empty `ep` → `terminate()`.
- Else restore the slot's `{type_id, object}` into active state, mark
  propagating, `++count`, and transfer control like a normal throw — via the
  raise primitive (§6).

Current gap: **not implemented** — `current_exception()` returns an empty
`exception_ptr` and `rethrow_exception()` is a hard `assert(0, "not modelled
yet")` (`src/cpp/library/exception`). This is §8 live work, steps 1–2.

### 4.5 `uncaught_exception()` / `uncaught_exceptions()` — **implemented**
- Backed by the per-thread `$esbmc_exc_uncaught_count` the lowering maintains
  (++ at throw/rethrow, −− at handler entry): `uncaught_exception()` is
  `count != 0`, `uncaught_exceptions()` is `count`. The OM reads the global
  (`c2goto/library/exception.cpp`).
- Lowered-path-only by nature — the count is bookkeeping the lowering owns. The
  dynamic-spec replacement-throw path is handled (§4.2).
- Remaining caveat: not faithful inside a destructor running mid-unwind
  (§8 step 3).

---

## 5. Soundness constraints (from design review — must-fix)

These change the design; they are not optional polish.

### 5.1 The raise primitive is necessary but NOT sufficient
A bare `__ESBMC_raise(type_id, object)` is a *control-transfer* op. It does not
materialize objects or guarantee lifetime. Specifically:
- **`std::bad_cast`** from a failed reference `dynamic_cast<T&>` must throw a
  *materialized* `bad_cast` object (user code may `catch` by reference and call
  `what()` / inspect identity). A type-id alone is **unsound**. Current state:
  `remove_exceptions` rewrites the `__ESBMC_throw_bad_cast` intrinsic into a real
  `THROW` and constructs a `bad_cast`, but a *message-less* one — full object
  materialization (so `what()` and identity are faithful) is still owed here.
- **`std::throw_with_nested`** must synthesize a type deriving from both `T` and
  `nested_exception` and capture `current_exception()` at construction. `raise`
  cannot do this alone.
- **`rethrow_exception`** needs a lifetime-stable object in the slot, not a
  shallow pointer into thread-local active state.
- The "object" stored in active/slot state must be a **semantically constructed
  exception object**, not raw bytes / a shallow pointer — otherwise catch-by-
  reference and destructor side effects are under-modeled (unsound).

→ Split "materialize exception object" from "raise/transfer." Faithful
materialization is the remaining work (§8 step 1); the bespoke symex paths are
already gone, so the lowering must carry full object semantics itself.

### 5.2 Slot publication must be atomic; identity permanent
See §3.3. Torn handle reads or reused slot indices make `rethrow_exception`
raise the wrong exception — a soundness bug, not imprecision.

### 5.3 Omitting refcount is sound only under never-reuse
Without refcount you lose exception-object **destruction timing** (destruction
when the last referring `exception_ptr` dies). If the exception's destructor has
visible side effects (or would itself trip `terminate`), "never free" suppresses
them — a behavioral gap. Acceptable for v1 under the never-reuse invariant, but
tracked; add refcount + destructor-at-zero when a test needs it.

### 5.4 `terminate()` classification
`assume(false)` is correct (non-returning). Classifying *every* `terminate` as an
assertion failure is a false alarm where termination is the *specified* behavior.
For `noexcept`-escape, reporting FAILED is correct (it is a bug) and matches
existing ESBMC policy. Decide explicitly whether to keep "terminate = property
violation" or introduce a distinct "expected termination" outcome.

### 5.5 Catch-entry timing and "current inside handler"
- The catch-parameter initialization (catch-by-**value** copy-ctor) can itself
  throw → `terminate`. Clear active state / `--count` **after** the parameter is
  bound, not before, or `current_exception()`/`uncaught_exceptions()` observe the
  wrong state during the copy.
- Do not clear active/handled state before invoking terminate/unexpected
  handlers — `current_exception()` must still return the provoking exception
  inside them.

---

## 6. Lowering responsibilities

### 6.1 State-update points
- **throw**: store active `{type_id, object}`, `++count`.
- **handler entry**: push active onto handled stack, clear active propagation
  state, `--count` — *after* the catch-parameter copy (§5.5).
- **bare `throw;`**: take top of handled stack, make active, `++count`.
- **catch exit**: pop handled stack.
- **`rethrow_exception`**: make pointed slot active, `++count`.

### 6.2 The unifying raise primitive
A single lowering-recognized op — `__ESBMC_raise(type_id, object)` (and a
by-handle variant for rethrow) — that `remove_exceptions` lowers exactly like a
frontend `THROW`: arm active state, `++count`, `goto` the enclosing region's
dispatch. This is the keystone for: `rethrow_exception`, the unexpected-handler
rethrow path, and (combined with object materialization, §5.1) `bad_cast` and
`throw_with_nested`. It is the only piece that must live below the OM, and it is
a lowering primitive — not the symex engine.

### 6.3 Spec enforcement
`noexcept`/`throw()` → epilogue assertion (already lowered; reads PR #5126
`exception_spec` metadata). Dynamic-spec recovery → §4.2.

---

## 7. Scope flags (correctness boundaries of the lone lowered path)

With the engine gone these are no longer "verify before deleting" — each is
either a live correctness obligation or a hard reject in production.

- **Handler matching is richer than `type_id` equality**: base-class / reference
  / cv-adjusted / pointer / `catch(...)`. The registry already does a subtype
  closure + ref/value-slice/pointer/void*/ellipsis, so it is not pure equality —
  but multiple-inheritance and cv/pointer corners still need the full test
  matrix.
- **Handled stack has no C++ semantic bound**: overflow must fail loudly (assert)
  or tie to the unwind bound; never silently truncate (wrong frame on deep
  nested rethrow = unsound). Lands with §8 step 2.
- **Destructor unwinding is inseparable from exceptions** and is the largest
  remaining gap (cf. PR #5132 — §8 step 3). Side effects during unwinding work,
  but the exception state seen by an unwinding destructor is not yet faithful.
  Stress suites must cover: destructor side effects during unwinding *with* a
  correct uncaught count, catch-by-value copy-ctor-throw, function-try-blocks,
  `noexcept`/dynamic-spec exits, deep nested rethrow.
- **`longjmp`/`setjmp`** needs an explicit semantic story (absence of the old
  engine's implicit interactions must not become a silent regression).
- **All frontends emitting THROW/CATCH** (Python) must satisfy the same lowering
  invariants; Python try/except/raise already shares the machinery.
- **Legacy dynamic specs** (`throw(T...)`): lowered (`build_dynamic_spec_check`),
  C++14-only and rejected under C++17. Functional, including the uncaught count
  across replacement-throw recovery (§4.2).
- **`dynamic_cast<T&>`**: the failed cast now reaches lowering as a real `THROW`
  rather than a symex-only `__ESBMC_throw_bad_cast` helper — keep it that way.

---

## 8. Build order and status

Completed (the engine deletion ran ahead of the original 1→7 sequence):

- **[done] dynamic_cast / `bad_cast`** (was step 4): the failed `dynamic_cast<T&>`
  `__ESBMC_throw_bad_cast` intrinsic is rewritten to a real `THROW` and lowered;
  the bespoke `symex_throw_bad_cast` path is gone. Object *materialization* for
  `bad_cast` is still the message-less form (§5.1 remains the deeper goal).
- **[done] dynamic-spec recovery** (was step 5): lowered `unexpected` /
  replacement-throw checking / `bad_exception` substitution
  (`build_dynamic_spec_check`); the `set_unexpected` builtin intercept is
  removed — the OM `__ESBMC_run_unexpected` runs the installed handler.
- **[done] terminate/unexpected routing** (was step 6): routed through the
  compiled OM (`std::terminate`/`std::unexpected`). Terminate-outcome
  classification (§5.4) is settled as "property violation" for `noexcept`-escape
  and "silent valid termination" (`assume(0)`) for `abort()`/`exit()`.
- **[done] engine deletion** (was step 7): `symex_catch.cpp` and the imperative
  plumbing removed; lowering is unconditional; `--lower-exceptions` flag, its
  differential gate script, and the design doc deleted.
- **[done] `uncaught_exception(s)`** (was step 3): a per-thread
  `$esbmc_exc_uncaught_count` is maintained by the lowering (++ at throw/rethrow,
  −− at handler entry, and a balancing −− on the dynamic-spec replacement-throw
  path) and read by the OM. **Remaining gap:** the count is not faithful inside a
  destructor running during unwinding (see the destructor item below).
- **[done] `exception_ptr`** (was step 2): a shared slot table plus a per-thread
  handled-exception stack in the OM (`c2goto/library/exception.cpp`);
  `current_exception` / `rethrow_exception` / `make_exception_ptr` work
  (`current_exception` reads the handled stack, rethrow re-arms from the slot).
  The handled-stack/uncaught instrumentation is pay-per-use (only emitted when a
  program uses current_exception / a bare throw / uncaught_exception(s)).
- **[done] catch-by-value / object materialization for the common case**
  (was step 1, partial): throwing a named/by-value operand now constructs the
  exception object correctly (the `cpp-throw` operand is adjusted so the
  copy/move ctor's reference args are addressed). `std::throw_with_nested` /
  `rethrow_if_nested` are implemented; they are blocked only by a frontend
  vtable-model bug (primary-base vptr not shared — see KNOWNBUG
  lower-exceptions_throw_with_nested_knownbug), not by the lowering.

Live work (still open):

1. **Faithful exception-object materialization** (the deeper §5.1 goal): a real
   `std::bad_cast` carrying `what()`/identity (today it is message-less), and
   `throw_with_nested` once the MI primary-base-vptr-sharing bug is fixed. The
   common-case construction is done (above); this is the remaining fidelity.

Newly-tracked gaps the deletion turned into hard rejects (no fallback remains):

3. **Destructor unwinding × exception state.** Destructor *side effects* during
   unwinding work (`throw_dtor_unwind_nested` passes), but the exception-state
   bookkeeping is not faithful inside an unwinding destructor — e.g.
   `uncaught_exceptions()` read in a dtor mid-unwind is wrong. This is §7's
   "largest gap" promoted to an explicit step; aligns with PR #5132.
4. **[done] Computed pthread start routine.** A `pthread_create` whose start
   routine is reached through a function pointer is no longer rejected: the
   lowering over-approximates by treating every address-taken void*(void*)
   function as a candidate thread entry (sound and complete for in-program
   routines, since a routine must have its address taken to be passed), so the
   uncaught→`terminate` boundary check still applies
   (`collect_unresolved_thread_routines`).
5. **[not a real gap] `--function` / no `__ESBMC_main` entry.** Verified: the C,
   C++ and Python frontends always generate `__ESBMC_main` (it wraps the
   `--function` target, even with no `main` in the TU), so the
   "no whole-program entry" check never fires in practice. The check is retained
   as a sound guard for any future entry-less mode (e.g. raw goto-binary input),
   where lowering without an anchor would be unsound.

Not feature gaps (defensive IR guards; no step required): a malformed catch
clause, an unmatched catch pop, and a throw of an *unsupported* type (the
`program_supported` reject "a throw of an unsupported type" — an empty
exception list or a thrown type the registry never ingested) are rejected as
malformed/unmodellable input. Note the asymmetry: an unregistered *catch* type
is **not** rejected — it is kept as a dead handler whose match guard is simply
false (`program_supported`), since no throw can select it. The old
"all-or-nothing leaves the rest to symex" property is moot — there is no symex
exception path to fall back to.

Possible later cleanup:

6. If a future need arises to move more `<exception>` behavior into a compiled OM
   and it reintroduces duplicate compiler-generated thunk symbols across TUs,
   teach the symbol linker / C++ merge path to coalesce them. The current narrow
   OM does not hit this, so it is not blocking.

---

## 9. Testing

The ON-vs-OFF differential gate is gone (there is no OFF path, and the
`lower_exceptions_differential.py` script and its workflow were deleted with the
flag). Correctness is now anchored on the regression suites directly:

- **Behavioural verdict validation**: the `regression/esbmc-cpp/try_catch` and
  `regression/python` exception suites are the source of truth; verdicts have
  been cross-checked against real `g++` behaviour for the C++ subset.
- **Standalone tests** for capabilities with no historical baseline:
  `uncaught_exception(s)` (done — `lower-exceptions_uncaught_count{,_fail}`),
  custom `set_terminate`/`set_unexpected` dispatch, `bad_exception` substitution;
  and, when implemented, `current_exception`/`rethrow_exception`.
- **Stress tests still owed** (§7): destructor side effects during unwinding
  *with* faithful exception-state (uncaught count, object lifetime); catch-by-
  value copy-ctor that throws; function-try-blocks; deep nested rethrow;
  multiple-inheritance handler selection; cross-thread `exception_ptr`
  capture/rethrow.
- **Frontend contract**: a failed `dynamic_cast<T&>` reaches lowering as an
  explicit `THROW` (done); keep this property under test so it never regresses to
  a symex-only helper.

---

## 10. Open decisions

1. **[settled]** Thread-local representation: the existing `$esbmc_exc_*` globals
   (now including the uncaught count) are marked `is_thread_local`; symex routes
   them per-thread. The `exception_ptr` slot table (§3.2) stays a shared global.
2. **[settled]** `terminate` outcome classification (§5.4): a `noexcept`-escape
   reports a property violation; an `abort()`/`exit()` from a terminate handler
   is a silent valid termination (`assume(0)`), per ESBMC convention.
3. Handled-stack bound and overflow behaviour (§7) — open, lands with §8 step 2.
4. Whether to model exception-object destruction timing in v1 (refcount) or defer
   (§5.3) — open, lands with §8 step 2.
5. `longjmp`/`setjmp` semantics and which frontends are in scope for the unified
   substrate — open.

---

## 11. Interaction with related PRs

- **#5126**: `THROW_DECL`→metadata + frame-boundary enforcement. The lowering
  *consumes* its `exception_spec` metadata (noexcept enforcement + dynamic-spec
  recovery set).
- **#5132** (destructor unwinding on the throw path): directly supplies the
  destructor semantics §8 step 3 needs; align.
- **#5148** (`std::bad_cast` under elaborated tag name): the lowering already
  synthesizes/looks up the `bad_cast` type across tag spellings; deeper object
  *materialization* is still §8 step 1.
