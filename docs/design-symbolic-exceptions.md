# Symbolic exception lowering (`--lower-exceptions`)

Status: **experimental, default OFF.** Tracks issue
[#5075](https://github.com/esbmc/esbmc/issues/5075).

## Motivation

Historically ESBMC resolves C++/Python `throw`/`catch` *imperatively during
symbolic execution* (`src/goto-symex/symex_catch.cpp`): a runtime `stack_catch`
of type-name→handler-target maps is consulted on each `THROW`, picking one
handler by **string comparison**. This is fragile — it commits to a single
target, matches on type strings rather than the thrown object's symbolic type,
mishandles nested propagation (a throw is only matched against the innermost
try, never propagated outward within a function), and has produced segfaults on
catch-all handlers (PR #5070).

`--lower-exceptions` instead **lowers throw/catch into ordinary guarded control
flow before symbolic execution**, so dispatch becomes guards the SMT solver
reasons about. The imperative path is untouched and remains the default.

## Architecture

A goto-functions transformation, `remove_exceptions`
(`src/goto-programs/remove_exceptions.cpp`), runs after
`remove_no_op`/`remove_unreachable` and **before** `goto_partial_inline`
(`src/esbmc/esbmc_parseoptions.cpp`).

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

Whole-program, **all-or-nothing**: lowered and imperative dispatch cannot
interoperate across a call, so unless every function is in the supported subset
(and a `main` exists) the pass lowers nothing. Per function it:

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

## Supported subset (current)

Lowered: C++ class **and primitive** throws (`throw 1`); reference, value (incl.
single-inheritance base-by-value slicing), pointer (`catch (T*)`) and `void*`
catches, and catch-all; nested try with inner→outer propagation; inter-procedural
propagation through direct and indirect calls; rethrow; uncaught detection at both
`main` and `__ESBMC_main` (the latter covers exceptions from global constructors
during static initialisation). Reaching outside the subset makes the whole program
fall back to the imperative path. About **39 of 67** `regression/esbmc-cpp/try_catch`
CORE tests currently lower, at 0 differential divergences.

**Python** lowers too: try/except/raise share the same THROW/CATCH machinery, the
registry ingests Python exception ancestry from THROW `exception_list`s, and the
entry/uncaught anchor is `__ESBMC_main` (which wraps `python_user_main`). 26 of 29
`regression/python/exception*`/`try_finally*`/`try_else*` tests lower at 0
divergences (the rest fall back).

Not yet lowered (fall back): parts of the `std` exception surface;
`bad_cast` from `dynamic_cast<T&>`; dynamic exception specifications /
`noexcept` (a throw inside a `THROW_DECL` spec region forces fallback).

**Destructor unwinding** is handled at the GOTO frontend (`convert_throw`), not
in the lowering pass, so it applies on **both** the imperative and lowered
paths: a throw runs the destructors of the automatic objects between the throw
point and the nearest enclosing try block ([except.ctor]), excluding the thrown
object itself. (Multi-level inner→outer propagation of those destructors within
a function relies on the lowered path's nested dispatch.)

`std::set_terminate` / `std::set_unexpected` are honoured: the operational model
(`src/cpp/library/exception`) now stores the installed handler and `terminate()`
/ `unexpected()` dispatch to it (previously stubbed — handlers were silently
ignored on all paths).

## Roadmap to default-on

The imperative path can only be removed once the lowered path reaches parity.
Remaining work, roughly ordered: throw-spec / `noexcept` (a throw escaping the
spec'd function should route to terminate rather than force fallback); the broader
`std` exception surface; `bad_cast` from `dynamic_cast<T&>`.
Then: two green full-suite differential runs (`--lower-exceptions` ON vs OFF)
before flipping the default and deleting `symex_catch.cpp`.

## Testing

`regression/esbmc-cpp/try_catch/lower-exceptions_*` exercise the lowered path
(simple, value-fail, nested, uncaught, rethrow, inter-procedural, indirect-call,
value-catch, slice, primitive-fallback). `unit/goto-programs/exception_typeid.test.cpp`
covers the registry. The development gate is differential equivalence
(ON vs OFF) across `regression/esbmc-cpp/try_catch`.
