---
title: C++ Limitations
---

> **Note**: The limitations below apply to the current version of ESBMC's C++
> frontend. Many are actively being addressed — check the
> [`C++` issue label](https://github.com/esbmc/esbmc/issues?q=is%3Aissue+is%3Aopen+label%3AC%2B%2B)
> for the latest status. For what *is* supported, see
> [C++ Support](/docs/c-cpp/supported-features).

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
- **`deque(iterator, iterator)` has no allocator-taking form.** Every other
  `list` and `deque` constructor now pairs with one ([list.cons], [deque.cons]),
  so `std::list<int, A> c(a)` parses. The iterator-pair `deque` constructor is
  held back because its size has a separate off-by-one: accepting the standard
  spelling would turn a compile error into a silently wrong answer.
- **Comparing two `std::list` iterators with `<` is rejected**, as it is against
  libc++: a list iterator is not random-access, so the ordering has to come from
  a user-declared `operator<`. What changed is that such a user-declared
  operator is now *found* by argument-dependent lookup.
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

Base-subobject displacements — the override thunk adapting a `Base*` receiver,
both arms of `dynamic_cast`, and derived-to-base conversions under a virtual
base — are now taken from ESBMC's own class layout rather than Clang's
`ASTRecordLayout`, which disagreed with it under the Itanium primary-base rule
([#3894](https://github.com/esbmc/esbmc/issues/3894)). A hierarchy containing a
virtual base keeps a flattened layout that cannot express every shape; the
remaining ones are pinned as `KNOWNBUG` under
`regression/esbmc-cpp/inheritance/`. Some
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
  [Not modelled](/docs/c-cpp/supported-features#not-modelled) — `<ranges>`,
  `<format>`, `<print>`, `<coroutine>`, `<generator>`, `<barrier>` and
  `<stdfloat>` — have no operational model. `--mix-cpp-host-headers` lets an unmodelled include fall
  through to your host header, but ESBMC then has to verify the real
  implementation, which is frequently intractable.
- Three modelled headers decline to compute what they describe, so a property
  that depends on the result cannot be proved: `<regex>` answers every match
  nondeterministically, `<syncstream>` transfers no characters, and `<codecvt>`
  performs no conversion. `<charconv>` has the integral overloads only, and
  `<concepts>` omits the cross-type `_with` concepts and the `invocable` family
  rather than approximate them.
- `<version>` promises only the eight feature-test macros ESBMC can honour. A
  program that branches on a macro it leaves undefined takes the
  feature-unavailable path, which is the conservative answer but not always the
  one a host build takes.
- `<latch>`, `<semaphore>` and `<stop_token>` leave out the `constexpr`
  constructors, the timed acquires (there is no clock to time them against) and
  `jthread`.
- Mixing the operational models with host headers can produce ambiguity errors
  for names defined by both, such as `char_traits` and `istream`
  ([#3387](https://github.com/esbmc/esbmc/issues/3387)).
- The operational models are deliberately simplified abstractions of the real
  library ([#965](https://github.com/esbmc/esbmc/issues/965)). They are written
  for verification tractability, so their performance characteristics and
  internal representations do not match a production standard library.
- `std::filesystem::directory_iterator` yields a bounded, nondeterministic
  number of synthesised entries — nothing reads a real filesystem, so symbolic
  execution terminates. The cap is an *assumption* rather than an assertion, so
  a defect needing more entries than it allows is excluded silently, where the
  container models assert instead and report `capacity exceeded`.
- `std::ilogb`, `std::logb` and `std::nexttoward` resolve as overloads but have
  no model in ESBMC's libc, so they return a nondeterministic value rather than
  the C99 result.

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

`std::ratio` is complete — `<ratio>` carries [ratio.syn] in full, including
`ratio_add`, `ratio_subtract`, the six comparison aliases with their C++17 `_v`
variables, and the SI typedefs from `atto` to `exa`.

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
  A target clang no longer maps — `--ppc-macos`, whose triple it rejects as
  unknown — is reported as `PARSING ERROR` carrying clang's own diagnostic,
  rather than crashing ESBMC.
