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

- **`insert` on a nested `std::vector` does not converge under an unbounded
  strategy.** `std::vector<std::vector<int>>` now constructs its elements
  correctly, and `push_back` converges under `--incremental-bmc`, but the
  element copy-constructor loop that `insert`'s shift runs unwinds indefinitely.
  Use a bounded `--unwind N` run for that case.
- Some STL container regression tests remain marked `KNOWNBUG`
  ([#4400](https://github.com/esbmc/esbmc/issues/4400)); the `regression/esbmc-cpp`
  suites are the authoritative record of which specific cases fail.

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

## Expressions

An lvalue conditional over reference *variables* — `(c ? ra : rb).x = 2` —
aborts the frontend, and a conditional over pointers to members is unsupported
([#6717](https://github.com/esbmc/esbmc/issues/6717)). The reference-returning
call shape works.

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

## Time and clocks

`system_clock::now()` and `steady_clock::now()` read a shared counter that
advances by a non-negative nondeterministic step, so a reading is not wall-clock
time and the gap between two readings is unconstrained. Checking a program
against a particular instant needs an `__ESBMC_assume` on the value, and one
whose correctness depends on how much real time passed cannot be checked at all.
`system_clock::period` follows the target platform, so the range a `time_point`
represents — and the point at which it saturates — differs between Linux, Apple
and Windows.

Because the counter is shared, `system_clock` is monotone too, even though its
`is_steady` is false as the standard allows. A defect that needs the system
clock to jump backwards — an NTP correction, an operator resetting the clock —
is therefore outside what the model can produce.

Calendar and time-zone facilities are absent: the C++20 types
(`year_month_day`, `zoned_time`, `utc_clock`), the `chrono_literals` suffixes
(`10ms`), and the `floor` / `ceil` / `round` / `abs` duration helpers.
`std::this_thread::sleep_for` and `sleep_until` are absent too — ESBMC already
interleaves at every step, so a sleep would not constrain the schedule it
explores.

`std::ratio` covers what `<chrono>` needs and no more: `ratio_add`,
`ratio_subtract` and the comparison aliases (`ratio_equal`, `ratio_less`, …) are
not declared, nor are the SI aliases outside `nano` / `micro` / `milli`.

## Standard version

The default is C++17. C++20 and C++23 features require an explicit `--std` — for
example `--std c++20`. The operational models honour `--std` only where a
version guard has been added by hand — see
[Standard-version guards](/docs/c-cpp/supported-features#standard-version-guards) for the
headers that have one; the models are not systematically versioned
([#1678](https://github.com/esbmc/esbmc/issues/1678)).

## Bounded verification

These are properties of bounded model checking rather than of the C++ frontend,
but they shape what a C++ result means:

- Loops are unwound to a finite bound. A plain `--unwind N` run proves the
  absence of bugs only up to `N` iterations; use `--incremental-bmc` or
  `--k-induction` to seek an unbounded proof.
- ESBMC targets x86_64 by default; other target architectures are not yet
  generically supported ([#1585](https://github.com/esbmc/esbmc/issues/1585)).
