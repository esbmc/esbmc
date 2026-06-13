---
title: Function Contracts and Modular Verification
weight: 35
---

Contracts are the basis of **modular (assume-guarantee) verification** in ESBMC.
This page explains the underlying principle and how it fits the other
[verification algorithms](/docs/theory/verification-algorithms); for the full
annotation syntax, command-line flags, worked examples, and limitations, see the
[**Function Contracts** how-to guide](/docs/function-contracts).

## The assume-guarantee principle

A contract states what a function *assumes* of its callers (a precondition) and
what it *guarantees* in return (a postcondition), together with a frame
condition naming the locations it may modify. Given such a contract, a function
is verified **once** against its specification, and every call to it is then
replaced by that specification rather than re-analysed.

This decomposes a single whole-program proof into independent per-function
obligations:

- **Enforce** — assume the precondition, run the body, assert the postcondition
  and frame. This discharges "the implementation meets its contract."
- **Replace** — at a call site, assert the precondition, havoc the frame, and
  assume the postcondition. This lets a caller be verified against the
  *specification* of its callees, without unrolling their bodies.

Enforcing each function once and replacing it everywhere it is called yields a
proof whose cost does not blow up with call depth — the key scalability win over
inlining everything into one [bounded model checking](/docs/theory/verification-algorithms#bmc)
query.

## Relationship to the other algorithms

Function contracts are an *over-approximation*: replacing a call with its
contract admits any behaviour the postcondition permits, which may be more than
the concrete body produces. This is sound for proofs but, like any
abstraction, can yield spurious counterexamples if a contract is too weak —
tightening the `ensures`/`assigns` clauses recovers precision. (The how-to
guide's *Replace mode is an over-approximation* note covers this in practice.)

The same assume-guarantee idea applies to loops: a **loop invariant** plus a
loop frame condition lets ESBMC summarise a loop instead of unrolling it,
turning a bounded check into an inductive one — closely related to
[k-induction](/docs/theory/verification-algorithms#k-induction) and detailed in
the [Loop Invariants](/docs/loop-invariants) guide.

## See also

- [Function Contracts how-to guide](/docs/function-contracts) — syntax,
  `--enforce-contract` / `--replace-call-with-contract`, quantifiers,
  `__ESBMC_is_fresh`, and known limitations.
- [Loop Invariants](/docs/loop-invariants) — loop contracts and the frame rule.
