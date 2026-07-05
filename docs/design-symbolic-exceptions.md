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
(`src/esbmc/esbmc_parseoptions.cpp`). It runs unconditionally.

### Global exception state (`exception_globals.{h,cpp}`)

Three zero-initialised globals carry the in-flight exception:

- `__ESBMC_exc_thrown : _Bool` — is an exception propagating?
- `__ESBMC_exc_typeid : size_t` — dynamic type id of the thrown object.
- `__ESBMC_exc_value  : void*`  — pointer to a static copy of the object.

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
propagation through direct and indirect calls; rethrow; uncaught detection at both
`main` and `__ESBMC_main` (the latter covers exceptions from global constructors
during static initialisation); `noexcept` / `throw()` enforcement (an exception
escaping a no-throw function → terminate, asserted at its epilogue); and
**dynamic exception specifications** (`throw(T...)`): the epilogue check
generalises the noexcept one — an exception in flight whose type is not in the
specification's allowed set (`__ESBMC_exc_typeid ∈ { id(U) : U <: some listed T }`)
is a specification violation, reported as "not allowed by declaration"
(`std::unexpected` / handler dispatch is modelled on neither the old nor the new
path).

**Python** lowers too: try/except/raise share the same THROW/CATCH machinery, the
registry ingests Python exception ancestry from THROW `exception_list`s, and the
entry/uncaught anchor is `__ESBMC_main` (which wraps `python_user_main`). All
`regression/python` exception tests lower (including model-raised exceptions — a
`KeyError` from `del d[k]`, a `TypeError` from mutating a tuple, a `ValueError`
from `math.factorial(-1)` — and the common `try: <raises>; assert False; except
E:` idiom).

A failing `dynamic_cast<T&>` lowers to a call to the bodyless intrinsic
`__ESBMC_throw_bad_cast`; the pass rewrites that call into an ordinary `THROW` of
a synthesized `std::bad_cast` (`build_bad_cast_throw`) so the rest of the pipeline
lowers it like any other throw. If `<typeinfo>`'s `std::bad_cast` is not in the
symbol table the program is reported as unsupported (see below).

The **`std` exception hierarchy** lowers through the same machinery, with no
std-specific handling: the frontend flattens a thrown std type's base chain into
the `THROW` `exception_list` (e.g. `THROW std::runtime_error, std::exception`),
which `register_chain` ingests, so a `catch (std::exception&)` base handler's
guard matches. Throwing `std::bad_alloc` from `new`, calling `what()` in a
handler, and user types deriving from `std::exception` all lower. (Two orthogonal
caveats: a `std::string` exception message drives the unbounded `strlen` model,
so such programs need an `--unwind` bound; and the frontend can omit intermediate
bases from the flattened chain, so a catch on a mid-hierarchy base may not match —
a frontend chain-completeness matter, not a lowering one.)

**Destructor unwinding** is handled at the GOTO frontend (`convert_throw`), not in
the lowering pass: a throw runs the destructors of the automatic objects between
the throw point and the nearest enclosing try block ([except.ctor]), excluding the
thrown object itself.

`std::set_terminate` / `std::set_unexpected` are honoured: the operational model
(`src/cpp/library/exception`) stores the installed handler and `terminate()` /
`unexpected()` dispatch to it.

## Unsupported programs (reported as errors)

Because lowering is the only exception path, an exception-using program outside
the supported subset is reported as a hard error (`report_unsupported`, which
throws the ESBMC fatal-error idiom — a `std::string` caught by
`process_goto_program` — logging `exception lowering: cannot lower <construct>`
and stopping verification cleanly, rather than `abort()`/SIGABRT). The residual:

- **concurrent programs that use exceptions** — the exception state is one global
  tuple, not per-thread, so the lowered dispatch is unsound across threads (one
  thread could observe or clear another's in-flight exception); per-thread state
  is a separate, unimplemented feature;
- **`--function` isolated verification** of exception code (no `__ESBMC_main`
  whole-program entry, so an uncaught exception could be silently accepted);
- **`dynamic_cast<T&>` whose `std::bad_cast` is unavailable** (the program never
  pulls in `<typeinfo>`, so the failure-path intrinsic cannot be lowered);
- a few unusual shapes (a value-catch without a copy binding, e.g.
  `std::bad_exception` by value; a malformed catch clause).

An exception-**free** program in any of these categories is unaffected: the pass
is a silent no-op for it (`report_unsupported` returns early via the
`program_uses_exceptions` guard).

## Testing

`regression/esbmc-cpp/try_catch/lower-exceptions_*` exercise the lowered path
(simple, value-fail, nested, uncaught, rethrow, inter-procedural, indirect-call,
value-catch, slice, primitive-fallback, empty catch-all). The unsupported-program
errors are pinned by `lower-exceptions_concurrent` and
`lower-exceptions_bad_cast_unsupported` (each asserting the hard error), with
`lower-exceptions_concurrent_no_exc` as a positive control (an exception-free
concurrent program still verifies). `unit/goto-programs/exception_typeid.test.cpp`
covers the registry.

The removed branches were discharged with a Mode-C C-Dead proof: after
`remove_exceptions` runs, no `THROW`/`CATCH`/`THROW_DECL`/`THROW_DECL_END`
instruction survives to symex (confirmed by `--goto-functions-only` dumps under
both Bitwuzla and Z3); the `default:` case in symex's instruction switch
(`log_error` + `abort`) is the backstop that would fire visibly if one ever did.
