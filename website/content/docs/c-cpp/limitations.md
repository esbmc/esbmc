---
title: C++ Limitations
---

> **Note**: The limitations below apply to the current version of ESBMC's C++
> frontend. Many are actively being addressed — check the
> [`C++` issue label](https://github.com/esbmc/esbmc/issues?q=is%3Aissue+is%3Aopen+label%3AC%2B%2B)
> for the latest status. For what *is* supported, see
> [C++ Support](./supported-features).

## Constructor and destructor ordering

The order in which constructors and destructors run is not correct in every
case, tracked by the umbrella issue
[#940](https://github.com/esbmc/esbmc/issues/940). The affected cases are:

- Virtual destructors
- Base initialization for the most-derived class
- Order of destruction under object composition (part-whole relationships)

A program whose correctness depends on one of these orderings may verify when it
should not, or vice versa.

## Containers

- **A `std::vector` of a non-trivial element type** — `std::vector<std::vector<int>>`,
  or a vector of any type with a non-trivial constructor — fails, because
  elements are assigned into raw `malloc`'d storage without the constructor
  running ([#6368](https://github.com/esbmc/esbmc/issues/6368)). Prefer a
  fixed-size array, or a vector of a trivial type, where you can.
- Some STL container regression tests remain marked `KNOWNBUG`
  ([#4400](https://github.com/esbmc/esbmc/issues/4400)); the `regression/esbmc-cpp`
  suites are the authoritative record of which specific cases fail.

## Memory management

- **A replacement `operator new` / `operator delete` is never called.** A
  user-defined global or class-level replacement operator is ignored, so both
  missed bugs and false alarms are possible
  ([#6494](https://github.com/esbmc/esbmc/issues/6494)).
- **Placement `new` can report a spurious heap out-of-bounds** under
  `--incremental-bmc` and `--k-induction`
  ([#6464](https://github.com/esbmc/esbmc/issues/6464)). A fixed-bound run
  (`--unwind N`) avoids it.

## Exceptions

- Under `--lower-exceptions`, a `pthread` start routine reached through a
  computed function pointer — or one that is both called directly and used as a
  start routine — cannot be given a sound per-function uncaught-escape check,
  and is declined as unsupported. Declining is sound: ESBMC never validates a
  buggy program as a result, it reports that it cannot analyse it.
- Some exception-handling regression tests remain marked `KNOWNBUG`
  ([#4402](https://github.com/esbmc/esbmc/issues/4402)).

## Inheritance and polymorphism

Virtual dispatch through a non-first base under multiple inheritance relies on
Clang's `ASTRecordLayout` in a way that is known to be brittle
([#3894](https://github.com/esbmc/esbmc/issues/3894)). Some
inheritance/polymorphism regressions remain marked `KNOWNBUG`
([#4399](https://github.com/esbmc/esbmc/issues/4399)), as do some of the
`gcc-template-tests` ([#4398](https://github.com/esbmc/esbmc/issues/4398)).

## Standard library

- Headers listed under
  [Not modelled](./supported-features#not-modelled) — including `<regex>`,
  `<ranges>`, `<format>`, `<forward_list>` and `<coroutine>` — have no
  operational model. `--mix-cpp-host-headers` lets an unmodelled include fall
  through to your host header, but ESBMC then has to verify the real
  implementation, which is frequently intractable.
- Mixing the operational models with host headers can produce ambiguity errors
  for names defined by both, such as `char_traits` and `istream`
  ([#3387](https://github.com/esbmc/esbmc/issues/3387)).
- The operational models are deliberately simplified abstractions of the real
  library ([#965](https://github.com/esbmc/esbmc/issues/965)). They are written
  for verification tractability, so their performance characteristics and
  internal representations do not match a production standard library.

## Standard version

The default is C++17. C++20 and C++23 features require an explicit `--std` — for
example `--std c++20`. ESBMC does not currently take the standard into account
when parsing the C++ libraries themselves
([#1678](https://github.com/esbmc/esbmc/issues/1678)).

## Bounded verification

These are properties of bounded model checking rather than of the C++ frontend,
but they shape what a C++ result means:

- Loops are unwound to a finite bound. A plain `--unwind N` run proves the
  absence of bugs only up to `N` iterations; use `--incremental-bmc` or
  `--k-induction` to seek an unbounded proof.
- ESBMC targets x86_64 by default; other target architectures are not yet
  generically supported ([#1585](https://github.com/esbmc/esbmc/issues/1585)).
