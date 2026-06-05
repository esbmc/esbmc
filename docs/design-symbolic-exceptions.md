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
during static initialisation); and `noexcept` / `throw()` enforcement (an exception
escaping a no-throw function → terminate, asserted at its epilogue). Reaching
outside the subset makes the whole program fall back to the imperative path. About
**47 of 67** `regression/esbmc-cpp/try_catch` CORE tests currently lower, at 0
differential divergences.

**Python** lowers too: try/except/raise share the same THROW/CATCH machinery, the
registry ingests Python exception ancestry from THROW `exception_list`s, and the
entry/uncaught anchor is `__ESBMC_main` (which wraps `python_user_main`). 26 of 29
`regression/python/exception*`/`try_finally*`/`try_else*` tests lower at 0
divergences (the rest fall back).

A failing `dynamic_cast<T&>` lowers to a call to the bodyless intrinsic
`__ESBMC_throw_bad_cast`; the pass rewrites that call into an ordinary `THROW` of
a synthesized `std::bad_cast` (mirroring `goto_symext::symex_throw_bad_cast`) so
the rest of the pipeline lowers it like any other throw. If `<typeinfo>`'s
`std::bad_cast` is not in the symbol table the call is left in place and the
program falls back to the imperative path.

The **`std` exception hierarchy** lowers through the same machinery, with no
std-specific handling: the frontend flattens a thrown std type's base chain into
the `THROW` `exception_list` (e.g. `THROW std::runtime_error, std::exception`),
which `register_chain` ingests, so a `catch (std::exception&)` base handler's
guard matches. Throwing `std::bad_alloc` from `new`, calling `what()` in a
handler, and user types deriving from `std::exception` all lower at 0
differential divergences. (Two orthogonal caveats, both affecting the imperative
path equally: a `std::string` exception message drives the unbounded `strlen`
model, so such programs need an `--unwind` bound; and the frontend can omit
intermediate bases from the flattened chain, so a catch on a mid-hierarchy base
may not match — a frontend chain-completeness matter, not a lowering one.)

Not yet lowered (fall back): dynamic exception specifications with real types
(`throw(T...)`, a C++14-only form the frontend rejects under C++17, so it forces
fallback only under `--std c++14`).

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

The main exception constructs now lower (class/primitive/std throws, the catch
forms, propagation, rethrow, noexcept, bad_cast); the remaining gate is an
exhaustive full-suite `--lower-exceptions` ON-vs-OFF differential (across all C++
and Python suites, not just `try_catch`) to surface any residual divergence,
after which the default can be flipped and `symex_catch.cpp` deleted. That gate
is automated by `scripts/lower_exceptions_differential.py` and the
`lower-exceptions-differential` GitHub Actions workflow: for every
exception-bearing regression test it runs ESBMC with and without the flag (the
exact command `regression/testing_tool.py` would build) and fails on any verdict
divergence. Two green full-suite runs are required before the flip.

## Testing

`regression/esbmc-cpp/try_catch/lower-exceptions_*` exercise the lowered path
(simple, value-fail, nested, uncaught, rethrow, inter-procedural, indirect-call,
value-catch, slice, primitive-fallback). `unit/goto-programs/exception_typeid.test.cpp`
covers the registry. The development gate is differential equivalence
(ON vs OFF), automated by `scripts/lower_exceptions_differential.py` — run it
locally as

```sh
scripts/lower_exceptions_differential.py --esbmc build/src/esbmc/esbmc
```

to diff every CORE exception-bearing C++ and Python test that asserts a
`VERIFICATION` verdict (tests expecting a frontend/parse error or an
unsupported-feature message are excluded — the post-GOTO lowering pass cannot
affect them), or narrow with `--root regression/esbmc-cpp/try_catch`. CI runs
the same gate via the `lower-exceptions-differential` workflow on a clean Linux
runner. The first full-suite run is green: 0 divergences across the comparable
C++ and Python exception corpus.
