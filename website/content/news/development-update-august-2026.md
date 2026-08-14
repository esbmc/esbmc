---
title: "Development Update: August 2026"
date: 2026-08-13T20:30:00+01:00
draft: false
tags:
  - ESBMC
  - FormalVerification
  - ModelChecking
  - OpenSource
---

Development on ESBMC has been moving fast: around 80 pull requests were
merged into `master` in the first two weeks of August alone. Here are the
highlights.

**A user-facing `esbmc.h` header.**
[#6932](https://github.com/esbmc/esbmc/pull/6932) ships `include/esbmc.h`,
installed alongside the binary, which exposes the verification intrinsics under
unprefixed names — `ESBMC_assume`, `ESBMC_assert`, `ESBMC_nondet_int`, and
friends. Including the header under any other compiler is a hard error, so
verification harnesses cannot silently compile to no-ops.

**Function contracts come to Python.**
[#6940](https://github.com/esbmc/esbmc/pull/6940) lowers `__ESBMC_requires` /
`__ESBMC_ensures` clauses in the Python frontend, so `--enforce-contract` and
`--replace-call-with-contract` now work on Python functions, with
`__ESBMC_return_value` typed from the function's return annotation. On the
C/C++ side, `__ESBMC_old` now insists on an lvalue
([#6917](https://github.com/esbmc/esbmc/pull/6917)), `__ESBMC_is_fresh`
callers are held to the extent they asked for
([#6919](https://github.com/esbmc/esbmc/pull/6919)), and contract-clause
misuse is diagnosed instead of silently lifted
([#6944](https://github.com/esbmc/esbmc/pull/6944),
[#6980](https://github.com/esbmc/esbmc/pull/6980)).

**Sounder, leaner concurrency exploration.**
[#6911](https://github.com/esbmc/esbmc/pull/6911) fixes a `--state-hashing`
soundness bug that collided states differing only in the active thread — four
regression tests return to their expected FAILED verdicts — and introduces
optional sleep sets (`--sleep-sets`) plus per-run counters reporting what each
schedule reduction prunes. [#6968](https://github.com/esbmc/esbmc/pull/6968)
tightens MPOR by checking its dependency chain points forward in time, and
[#6939](https://github.com/esbmc/esbmc/pull/6939) /
[#6947](https://github.com/esbmc/esbmc/pull/6947) compose schedule
falsification with unwinding strategies for faster bug-finding on concurrent
benchmarks.

**Memory layout faithful to the ABI.**
The C frontend now honours `#pragma pack` when laying out struct members
([#6955](https://github.com/esbmc/esbmc/pull/6955)), declaration alignment
survives type-symbol squashing
([#6956](https://github.com/esbmc/esbmc/pull/6956)), and the SMT backends give
non-packed objects the ABI's fundamental alignment
([#6957](https://github.com/esbmc/esbmc/pull/6957)). The pointer analysis also
stores whole floating-point elements directly instead of stitching bytes
([#6977](https://github.com/esbmc/esbmc/pull/6977)).

**More compiler builtins modelled.**
`__builtin_ffs/ffsl/ffsll`
([#6943](https://github.com/esbmc/esbmc/pull/6943),
[#6946](https://github.com/esbmc/esbmc/pull/6946)) and
`__builtin_clzg/ctzg` together with the `ctz` family
([#6928](https://github.com/esbmc/esbmc/pull/6928)) are now understood by
symbolic execution, and a user's own `abs` definition wins over the builtin
([#6906](https://github.com/esbmc/esbmc/pull/6906)).

**Python frontend.**
NumPy support grows substantially:
[#6909](https://github.com/esbmc/esbmc/pull/6909) extends `dot`, `transpose`,
`flatten`, `sum`, `mean`, `min`, `max`, `prod`, `std` and `var` to arrays built
by any constructor (`zeros`, `ones`, `full`, `eye`, `identity`, `linspace`,
`arange`), in both module and method form. Objects are now truth-tested through
`__bool__` in `not` and `assert`
([#6915](https://github.com/esbmc/esbmc/pull/6915)), and constructor
temporaries run `__init__` and dispatch their own methods correctly
([#6890](https://github.com/esbmc/esbmc/pull/6890),
[#6892](https://github.com/esbmc/esbmc/pull/6892),
[#6903](https://github.com/esbmc/esbmc/pull/6903)).

**C++ operational models.**
New coverage includes `<cwchar>` — the largest single blocker to
self-verification ([#6869](https://github.com/esbmc/esbmc/pull/6869)) —
`std::visit` ([#6918](https://github.com/esbmc/esbmc/pull/6918)),
`std::invoke_result` ([#6874](https://github.com/esbmc/esbmc/pull/6874)),
reverse range accessors ([#6872](https://github.com/esbmc/esbmc/pull/6872))
and `ostringstream` buffering fixes
([#6864](https://github.com/esbmc/esbmc/pull/6864)). Lambdas inside template
instantiations now get their own closure type
([#6976](https://github.com/esbmc/esbmc/pull/6976)).

**Java/Kotlin frontend modernisation.**
A sustained series of about twenty pull requests
([#6851](https://github.com/esbmc/esbmc/pull/6851)–[#6891](https://github.com/esbmc/esbmc/pull/6891))
migrates the Jimple frontend to build statements and expressions natively in
ESBMC's IREP2 representation — assignments, gotos, invocations, allocations,
array accesses and more — shedding a whole legacy conversion layer.

**Robustness.**
Deeply nested expressions are now reported as a graceful error instead of
crashing the process with stack exhaustion
([#6910](https://github.com/esbmc/esbmc/pull/6910),
[#6974](https://github.com/esbmc/esbmc/pull/6974)), a `malloc` whose result is
discarded still allocates — so leak checking sees it
([#6937](https://github.com/esbmc/esbmc/pull/6937)) — and the expression
simplifier gained missing peephole identities and an unsigned-negation
modulus fix ([#6931](https://github.com/esbmc/esbmc/pull/6931),
[#6933](https://github.com/esbmc/esbmc/pull/6933),
[#6975](https://github.com/esbmc/esbmc/pull/6975)).

As always, the full list is in the
[commit history](https://github.com/esbmc/esbmc/commits/master/) — and if you
hit a bug or a missing feature, we would love an
[issue report](https://github.com/esbmc/esbmc/issues).
