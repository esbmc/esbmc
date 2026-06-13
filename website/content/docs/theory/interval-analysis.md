---
title: Interval Analysis
weight: 30
---

Interval analysis is an [abstract interpretation](https://en.wikipedia.org/wiki/Abstract_interpretation)
ESBMC can run *before* SMT solving. It infers a conservative range for each
integer and floating-point variable at each program point and injects those
ranges back into the program as assumptions, shrinking the state space the
solver has to explore.

## How it works

```sh
esbmc file.c --interval-analysis
```

The analysis computes, for every variable, an interval `[lo, hi]` that
over-approximates the set of values the variable can hold at a given location.
Because it is an over-approximation, the inferred bounds are always sound: the
real value is guaranteed to lie within the interval, though the interval may be
wider than necessary.

ESBMC then adds the discovered facts to the program as `__ESBMC_assume`
constraints. These assumptions cannot remove real behaviours (they only restate
what the abstract domain already proved), but they give the SMT backend tighter
bounds up front — often pruning infeasible paths and resolving some properties
without a full bit-precise search. Interval analysis composes with the
[verification algorithms](/docs/theory/verification-algorithms): the assumptions
it injects are present in the BMC, falsification, incremental-BMC, and
k-induction encodings alike, and tighter ranges can in particular help
k-induction's inductive step converge.

## Tuning the domain

The base analysis covers linear behaviour; several flags extend or instrument
it:

- `--interval-analysis-arithmetic` — track ranges through arithmetic operations
- `--interval-analysis-bitwise` — reason about bitwise operations
- `--interval-analysis-modular` — model wrap-around (modular) integer arithmetic
- `--interval-analysis-wrapped` — use wrapped (modular) interval domains
- `--interval-analysis-simplify` — simplify the program using inferred intervals
- `--interval-analysis-extrapolate` / `--interval-analysis-narrowing` — widening
  and narrowing to make fixpoint computation over loops terminate and then
  recover precision

Use `--interval-analysis-dump` (or `--interval-analysis-csv-dump`) to print the
intervals the analysis inferred, which is useful for understanding why a path
was or was not pruned.

## Related: the goto contractor

```sh
esbmc file.c --goto-contractor
```

The goto contractor is a related, constraint-propagation technique: rather than
only forward-propagating ranges, it *contracts* variable domains against the
assertion condition at the GOTO level (using interval contractors), tightening
the values that could violate a property. It targets the assertions directly,
whereas `--interval-analysis` strengthens the whole program.
