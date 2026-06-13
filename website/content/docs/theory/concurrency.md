---
title: Concurrency and Context-Bounded Model Checking
weight: 25
---

The "CB" in ESBMC stands for **context-bounded**: when verifying multi-threaded
programs, ESBMC explores thread interleavings up to a bounded number of context
switches. This page explains how concurrent programs are modelled and which
properties are checked.

## Interleavings as symbolic schedules

ESBMC models a concurrent program (POSIX threads, and the language frontends
that lower onto them) as a set of threads whose instructions interleave.
Verification explores the reachable **schedules** — orderings of the threads'
visible operations — and for each schedule it builds the usual
[SSA / SMT encoding](/docs/theory/smt-formula-generation) and checks every
safety property. A property is violated if *some* interleaving reaches a bad
state.

The number of interleavings grows combinatorially with the number of threads
and operations, so ESBMC controls the explosion in three complementary ways.

## Bounding the context switches

```sh
esbmc file.c --context-bound K
```

A *context switch* is a point where execution passes from one thread to another.
`--context-bound K` limits each thread to at most *K* switches, restricting the
search to schedules with few preemptions. This is the context-bounding idea: in
practice, many concurrency bugs manifest within a small number of context
switches, so a small bound finds them cheaply while keeping the formula
tractable. The default is unbounded (`-1`).

## Partial-order reduction

Many interleavings differ only in the order of operations that do not interact
(for example, two threads touching disjoint memory) and therefore lead to
equivalent states. **Partial-order reduction (POR)** prunes such redundant
schedules, exploring one representative per equivalence class. POR is on by
default; disable it with `--no-por` (for example, to cross-check that the
reduction is not hiding a schedule).

## State hashing

```sh
esbmc file.c --state-hashing
```

State hashing records a fingerprint of each explored state and skips
re-exploring states already seen, pruning duplicate work across interleavings.

By default ESBMC stops at the first interleaving that violates a property; pass
`--all-runs` to keep checking the remaining interleavings even after a bug is
found.

## Atomic blocks

The modeling primitives `__ESBMC_atomic_begin()` / `__ESBMC_atomic_end()` mark a
region that executes without interruption from other threads, suppressing
context switches inside it (see
[Modeling with Non-determinism](/docs/theory/non-determinism)).

## Concurrency properties

Beyond the usual assertions and memory-safety checks (which apply per
interleaving), ESBMC offers concurrency-specific checks:

| Check | Flag |
|---|---|
| Data races (unsynchronised conflicting accesses to shared state) | `--data-races-check` |
| Deadlock (global and local, over mutexes) | `--deadlock-check` |
| Lock-acquisition ordering | `--lock-order-check` |
| Atomicity at visible assignments | `--atomicity-check` |

`--data-races-check-only` narrows the run to race checks to reduce overhead.

## Soundness note

The bounds above are search-space reductions, not approximations of the
semantics within the explored schedules: a counterexample ESBMC reports is a
real interleaving. A clean result, however, is relative to the chosen context
bound — increasing `--context-bound` (or removing it) explores deeper schedules
at higher cost.
