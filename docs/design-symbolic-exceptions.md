# Symbolic exception lowering

Status: **the only exception path.** Tracks issue
[#5075](https://github.com/esbmc/esbmc/issues/5075). ESBMC lowers C++/Python
`throw`/`catch` into ordinary guarded control flow before symbolic execution; the
legacy imperative path that resolved exceptions *during* symex
(`src/goto-symex/symex_catch.cpp`) has been removed. A program the lowering pass
cannot handle is reported as a hard error rather than miscompiled.

## Motivation

ESBMC historically resolved C++/Python `throw`/`catch` *imperatively during
symbolic execution* (the former `src/goto-symex/symex_catch.cpp`): a runtime
`stack_catch` of type-name→handler-target maps was consulted on each `THROW`,
picking one handler by **string comparison**. This was fragile — it committed to
a single target, matched on type strings rather than the thrown object's symbolic
type, mishandled nested propagation (a throw was only matched against the
innermost try, never propagated outward within a function), and produced
segfaults on catch-all handlers (PR #5070).

Lowering instead **rewrites throw/catch into ordinary guarded control flow before
symbolic execution**, so dispatch becomes guards the SMT solver reasons about.
The imperative path is gone; symex sees only assignments, gotos, asserts, and
function calls.

## Architecture

A goto-functions transformation, `remove_exceptions`
(`src/goto-programs/remove_exceptions.cpp`), runs after
`remove_no_op`/`remove_unreachable` and **before** `goto_partial_inline`
(`src/esbmc/parseoptions/process_goto_program.cpp`). It runs unconditionally.

### Exception state (`exception_globals.{h,cpp}`)

Five zero-initialised globals carry the in-flight exception. They are
**thread-local**: a propagating exception, its type and its object belong to the
thread that raised it, and symex routes thread-local globals to a per-thread
instance (`renaming.cpp`), so one thread cannot observe, catch, or clear
another's in-flight exception. That is what makes the lowered dispatch sound
under concurrency.

- `__ESBMC_exc_thrown : _Bool` — is an exception propagating?
- `__ESBMC_exc_typeid : size_t` — dynamic type id of the thrown object.
- `__ESBMC_exc_value  : void*`  — pointer to a static copy of the object.
- `__ESBMC_exc_uncaught_count : size_t` — exceptions thrown (or rethrown) in this
  thread that have not yet entered their matching handler; backs
  `std::uncaught_exception(s)` ([except.uncaught]).
- `__ESBMC_exc_terminate_reason : size_t` — which terminate point routed into
  `std::terminate()` (generic / uncaught / noexcept / exception-spec /
  no-active-exception), so the default handler keeps the original diagnostic.

Three bodyless intrinsics support rethrow and the handled-exception stack:
`__ESBMC_push_handled_exception`, `__ESBMC_pop_handled_exception`,
`__ESBMC_rethrow_current_exception`.

### Type-id registry (`exception_typeid.{h,cpp}`)

Closed-world (ESBMC sees the whole program): every class type gets a stable
integer id and a reflexive-transitive subtype closure, built from the symbol
table's `bases` metadata and from THROW `exception_list` chains
(`register_chain`, for frontends like Python whose exception classes have no
`tag-` symbol). A `catch (C)` becomes the finite guard
`__ESBMC_exc_typeid ∈ { id(T) : T <: C }`; `catch (...)` is `thrown == true`.

### Lowering (`remove_exceptions.cpp`)

Whole-program, **all-or-nothing**: unless every function is in the supported
subset (and a `__ESBMC_main` entry exists) the program is reported as
unsupported. Per function it:

- recovers the try-region tree from positional `CATCH` push/pop;
- copies each thrown object into a static slot (`__ESBMC_exc_obj$N`) so it
  outlives the throwing frame;
- replaces `THROW` with: arm the globals, `goto` the enclosing region's dispatch
  block (or the epilogue);
- emits a per-region **dispatch block** (after the skip-handlers `goto`, so
  normal completion bypasses it) that branches to the first matching handler,
  else propagates to the parent region / epilogue;
- inserts `IF thrown GOTO dispatch/epilogue` after every may-throw call —
  including indirect (function-pointer / virtual) calls, conservatively treated
  as may-throw — giving inter-procedural propagation;
- binds handlers: reference catch `v = (T*)value`, value catch
  `v = *(T*)value` (copy/slice), clearing `thrown` on entry;
- `throw;` (rethrow) re-raises from the globals;
- asserts *uncaught* at `main`'s epilogue.

Exception specifications are **function metadata** (`goto_functiont::exception_spec`),
not instructions: the `THROW_DECL` / `THROW_DECL_END` instruction kinds were
removed (see the note on slots 19 and 20 in `goto_program.h`).

**Unclosed-region rebalancing.** `remove_unreachable` runs before this pass and
prunes the empty `CATCH` pop + skip-GOTO of a try whose body cannot complete
normally — the common `try: <op that raises>; assert False; except E:` idiom,
where a model- or user-raised throw makes the fall-through dead. That leaves the
`CATCH` push unbalanced, which the positional region recovery cannot pair.
`rebalance_removed_pops` re-inserts a synthetic pop + skip-GOTO before each
unclosed push's first handler, restoring the balanced shape.

**Elided skip-GOTO restoration.** When a try's handlers are all empty (e.g.
`catch (...) {}`), the frontend elides the skip-handlers GOTO after the `CATCH`
pop, because the jump would target the pop's own fall-through (a no-op). The
dispatch-block placement needs that GOTO, so `insert_elided_skip_gotos`
re-inserts a behaviour-neutral `GOTO <next>` after any pop that lacks one.

## Supported subset

Lowered: C++ class **and primitive** throws (`throw 1`); reference, value (incl.
single-inheritance base-by-value slicing), pointer (`catch (T*)`) and `void*`
catches, and catch-all; nested try with inner→outer propagation; inter-procedural
propagation through direct and indirect calls; rethrow; multiple inheritance
(catch-by-base rebinds to the correct base subobject offset); uncaught detection
at both `main` and `__ESBMC_main` (the latter covers exceptions from global
constructors during static initialisation); `noexcept` / `throw()` enforcement (an
exception escaping a no-throw function → terminate, asserted at its epilogue); and
**dynamic exception specifications** (`throw(T...)`): the epilogue check
generalises the noexcept one — an exception in flight whose type is not in the
specification's allowed set (`__ESBMC_exc_typeid ∈ { id(U) : U <: some listed T }`)
runs `std::unexpected` and re-checks, with `std::bad_exception` substitution when
the specification allows it ([except.throw]/8, [except.unexpected]), otherwise
reporting "not allowed by declaration".

**Concurrency.** Because the exception state is thread-local, concurrent programs
that use exceptions lower soundly. Two thread shapes are still declined (below).

**Standard-library surface.** `std::set_terminate` / `std::set_unexpected` are
honoured: the operational model (`src/cpp/library/exception`) stores the installed
handler and `terminate()` / `unexpected()` dispatch to it. `std::exception_ptr`,
`current_exception`, `make_exception_ptr` and `rethrow_exception` are modelled, as
is `std::uncaught_exception(s)` (backed by `__ESBMC_exc_uncaught_count`).
`std::throw_with_nested` / `std::rethrow_if_nested` lower with conforming
[except.nested]/6 semantics — including the no-op case when the argument's
*static* type is non-polymorphic.

A failing `dynamic_cast<T&>` lowers to a call to the bodyless intrinsic
`__ESBMC_throw_bad_cast`; the pass rewrites that call into an ordinary `THROW` of
a synthesized `std::bad_cast` (`build_bad_cast_throw`) so the rest of the pipeline
lowers it like any other throw. This works whether or not the program pulls in
`<typeinfo>`.

The **`std` exception hierarchy** lowers through the same machinery, with no
std-specific handling: the frontend flattens a thrown std type's base chain into
the `THROW` `exception_list` (e.g. `THROW std::runtime_error, std::exception`),
which `register_chain` ingests, so both root and mid-hierarchy base handlers match
(`throw std::range_error` selects `catch (std::runtime_error&)` over
`catch (std::exception&)`). Throwing `std::bad_alloc` from `new`, calling `what()`
in a handler, and user types deriving from `std::exception` all lower. (One
orthogonal caveat: a `std::string` exception message drives the unbounded `strlen`
model, so such programs need an `--unwind` bound.)

**Destructor unwinding** is handled at the GOTO frontend (`convert_throw`), not in
the lowering pass: a throw runs the destructors of the automatic objects between
the throw point and the nearest enclosing try block ([except.ctor]), excluding the
thrown object itself.

**Python** lowers too: try/except/raise share the same THROW/CATCH machinery, the
registry ingests Python exception ancestry from THROW `exception_list`s, and the
entry/uncaught anchor is `__ESBMC_main` (which wraps `python_user_main`). All
`regression/python` exception tests lower (including model-raised exceptions — a
`KeyError` from `del d[k]`, a `TypeError` from mutating a tuple, a `ValueError`
from `math.factorial(-1)` — and the common `try: <raises>; assert False; except
E:` idiom).

## Unsupported programs (reported as errors)

Because lowering is the only exception path, an exception-using program outside
the supported subset is reported as a hard error (`report_unsupported`, which
throws the ESBMC fatal-error idiom — a `std::string` caught by
`process_goto_program` — logging `exception lowering: cannot lower <construct>`
and stopping verification cleanly, rather than `abort()`/SIGABRT). The residual:

- **a thread with an unresolved start routine** (reached through a computed
  pointer) and **a thread start routine that is also called directly** by name.
  `is_entry` is a per-function property, but terminate-on-escape is only correct
  on the thread-entry edge, so neither shape gets a sound per-function
  uncaught-escape check. Declining is sound — it never validates a buggy program.
  A sound-but-imprecise residual remains: a routine that is a clean `&worker`
  thread entry *and* is also invoked through an indirect call elsewhere keeps
  `is_entry` on that path, so an exception the indirect caller would catch is
  over-reported as a terminate. Fixing that needs call-site-sensitive enforcement
  at the pthread trampoline, blocked until thread-local state propagates across
  its indirect call;
- **`--function` isolated verification** of exception code (no `__ESBMC_main`
  whole-program entry, so an uncaught exception could be silently accepted);
- a few unusual shapes: a value catch without a copy binding (e.g.
  `std::bad_exception` by value), an unsupported handler shape, a malformed or
  unmatched/trailing catch clause, a throw of an unsupported type.

An exception-**free** program in any of these categories is unaffected: the pass
is a silent no-op for it (`report_unsupported` returns early via the
`program_uses_exceptions` guard), and the thread-shape checks only fire when the
program actually throws or catches.

## Testing

`regression/esbmc-cpp/try_catch/lower-exceptions_*` (52 tests) exercise the
lowered path: simple, value-fail, nested, uncaught, rethrow, inter-procedural,
indirect-call, value-catch, slice, primitive-fallback, empty catch-all, multiple
inheritance, `exception_ptr` / `make_exception_ptr`, `uncaught_exceptions`
counting, static-init escape, `noexcept` and dynamic-spec enforcement,
`bad_exception` substitution, `std` base-class catches, and `dynamic_cast`
`bad_cast` with and without `<typeinfo>`. The declined thread shapes are pinned by
`lower-exceptions_concurrent_dualuse`, `lower-exceptions_thread_computed_routine`
and `..._fail`, with `lower-exceptions_concurrent`,
`lower-exceptions_concurrent_throw_decl` and `lower-exceptions_concurrent_no_exc`
as positive controls (concurrent programs that *do* lower).
`unit/goto-programs/exception_typeid.test.cpp` covers the registry.

The removed branches were discharged with a Mode-C C-Dead proof: after
`remove_exceptions` runs, no `THROW`/`CATCH` instruction survives to symex
(confirmed by `--goto-functions-only` dumps under both Bitwuzla and Z3); the
`default:` case in symex's instruction switch (`log_error` + `abort`) is the
backstop that would fire visibly if one ever did.
