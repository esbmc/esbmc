---
title: "Development Update: Late August 2026"
date: 2026-08-22T07:30:00+01:00
draft: false
tags:
  - ESBMC
  - FormalVerification
  - ModelChecking
  - OpenSource
---

A follow-up to the [previous update](/news/development-update-august-2026): another
175 or so pull requests landed on `master` between 13 and 22 August. Here are
the highlights.

**Every run names every property.**
[#7064](https://github.com/esbmc/esbmc/pull/7064) prints the per-property
`** Results:` block on every run, not only under `--multi-property`, and
`--result-only` no longer suppresses it — so `esbmc --result-only file.c` now
names what it checked. Properties a single-property run never separated are
reported `NOT CHECKED` rather than `PASSED`, since a run that stops at the
first violation cannot decide the rest.

**Python gets dynamic typing.**
A variable whose type diverges across an `if`/`else` is carried as a tagged
value instead of being forced to one branch's type. Building on [#6535](https://github.com/esbmc/esbmc/pull/6535), the
mechanism now covers reassignment and arithmetic
([#7018](https://github.com/esbmc/esbmc/pull/7018)), `isinstance`
([#7117](https://github.com/esbmc/esbmc/pull/7117),
[#7179](https://github.com/esbmc/esbmc/pull/7179)), `is` and `==` comparisons
([#7135](https://github.com/esbmc/esbmc/pull/7135)), rebinding to a container
([#7178](https://github.com/esbmc/esbmc/pull/7178)) and function return values
([#7162](https://github.com/esbmc/esbmc/pull/7162)) — most of which previously
aborted the frontend or crashed symbolic execution outright.

**Contracts frame what they name.**
An `__ESBMC_assigns` clause naming an array now frames the whole array rather
than element `[0]` ([#7019](https://github.com/esbmc/esbmc/pull/7019)), a
multi-level path such as `o->sub->a` is rooted at its parameter instead of
generating zero verification conditions
([#7068](https://github.com/esbmc/esbmc/pull/7068)), and the element-level frame
rule was corrected to evaluate its index in the pre-state
([#7069](https://github.com/esbmc/esbmc/pull/7069),
[#7107](https://github.com/esbmc/esbmc/pull/7107),
[#7217](https://github.com/esbmc/esbmc/pull/7217)). `__ESBMC_old` works under a
quantifier ([#6964](https://github.com/esbmc/esbmc/pull/6964)), a named object
satisfies `__ESBMC_is_fresh`
([#7036](https://github.com/esbmc/esbmc/pull/7036)), and a replaced call's
`ensures` now speaks about the value the call returned
([#7033](https://github.com/esbmc/esbmc/pull/7033)). On the Python side,
`__ESBMC_old` is lowered for scalars
([#7038](https://github.com/esbmc/esbmc/pull/7038)), completing the scalar
contract MVP ([#6942](https://github.com/esbmc/esbmc/pull/6942)).

**A large C++ standard-library push.**
Around fifty operational-model pull requests landed. New this fortnight:
`std::reference_wrapper` with `ref`/`cref`
([#7147](https://github.com/esbmc/esbmc/pull/7147),
[#7213](https://github.com/esbmc/esbmc/pull/7213)),
`std::unordered_multiset` ([#7156](https://github.com/esbmc/esbmc/pull/7156)),
`<chrono>` clocks and `time_point`
([#6985](https://github.com/esbmc/esbmc/pull/6985)), `std::put_time` and
`std::placeholders` ([#7132](https://github.com/esbmc/esbmc/pull/7132)),
`filesystem::u8path` and the `path` string accessors
([#7169](https://github.com/esbmc/esbmc/pull/7169),
[#7180](https://github.com/esbmc/esbmc/pull/7180)), and a long list of
`<type_traits>` additions. The containers took their `Allocator` template
parameter ([#7163](https://github.com/esbmc/esbmc/pull/7163),
[#7167](https://github.com/esbmc/esbmc/pull/7167)), `std::function`'s call
target is templated on its signature
([#7171](https://github.com/esbmc/esbmc/pull/7171)), and `basic_string`'s
`size`, `resize`, `max_size` and `rfind` were brought into line with the
standard ([#7145](https://github.com/esbmc/esbmc/pull/7145),
[#7175](https://github.com/esbmc/esbmc/pull/7175)).

**Nothing in `/tmp` for C and C++ runs.**
[#6517](https://github.com/esbmc/esbmc/pull/6517) serves the bundled clang
headers, C++ operational models and internal libc to clang from memory instead
of extracting them — 320 files and 7.1 MB of `/tmp` traffic per C run, 488 files
and 8.0 MB per C++ run, both now zero, with wall clock unchanged. The
operational models are compiled straight out of the same store
([#6766](https://github.com/esbmc/esbmc/pull/6766)) and the Python models moved
to their own blob ([#7058](https://github.com/esbmc/esbmc/pull/7058)). Python
and Solidity still extract, since a forked `python3` or `solc` can only read
real files.

**A bound on the whole stack.**
[#6970](https://github.com/esbmc/esbmc/pull/6970) adds `--total-stack-limit`,
which sums the live frames rather than bounding one at a time, so a recursion
whose individual frames each fit is caught. ESBMC's own operational models are
excluded, so the bound stays calibratable against a real stack budget.

**Soundness and precision.**
The pointer analysis resolves pointers an aggregate holds at an offset
([#6981](https://github.com/esbmc/esbmc/pull/6981)) and reports misaligned reads
from pointer arrays ([#6979](https://github.com/esbmc/esbmc/pull/6979)) — both
were false `SUCCESSFUL` results on ordinary C. A pointer whose object id is a
free variable can no longer be placed at address zero
([#6999](https://github.com/esbmc/esbmc/pull/6999)). `fmod`, `remainder` and
`remquo` lower to the solver's exact FP remainder instead of
`x - y*(int)(x/y)`, which returned 64.0 for `fmod(1e18, 3.0)`
([#7015](https://github.com/esbmc/esbmc/pull/7015)). Two backends that used to
end a run with no verdict now report one
([#7065](https://github.com/esbmc/esbmc/pull/7065),
[#7078](https://github.com/esbmc/esbmc/pull/7078)).

**Python type inference and NumPy.**
A bare `list` parameter, or an unannotated one, recovers its element type from
the call sites ([#7192](https://github.com/esbmc/esbmc/pull/7192),
[#7199](https://github.com/esbmc/esbmc/pull/7199)), including in an imported
module ([#7198](https://github.com/esbmc/esbmc/pull/7198)). `try`/`finally` runs
the cleanup on the `return`/`break`/`continue` edges
([#7181](https://github.com/esbmc/esbmc/pull/7181)), `len()` on a class without
`__len__` raises `TypeError` instead of measuring the struct with `strlen` and
answering 0 — which had silently emptied `for` loops
([#7131](https://github.com/esbmc/esbmc/pull/7131),
[#7137](https://github.com/esbmc/esbmc/pull/7137)) — and a generator's `if`
clauses survive being consumed by `sum`, `min`, `max` or `sorted`
([#7214](https://github.com/esbmc/esbmc/pull/7214)). In NumPy, `arange` is
materialised at conversion time
([#7020](https://github.com/esbmc/esbmc/pull/7020)), logical calls chain as
arguments to other calls ([#7089](https://github.com/esbmc/esbmc/pull/7089)),
and a 1-D unit-stride slice is a real view onto the base array rather than a
copy ([#7144](https://github.com/esbmc/esbmc/pull/7144)).

The website documentation has been updated to match. As always, the full list is
in the [commit history](https://github.com/esbmc/esbmc/commits/master/) — and if
you hit a bug or a missing feature, we would love an
[issue report](https://github.com/esbmc/esbmc/issues).
