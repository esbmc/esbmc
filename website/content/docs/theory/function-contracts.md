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
what it *guarantees* in return (a postcondition), together with a **frame
condition**. Given such a contract, a function is verified **once** against its
specification, and every call to it is then replaced by that specification
rather than re-analysed.

A *frame condition* names the locations an operation is allowed to modify and —
crucially — asserts that **every other location is left unchanged**. In a
contract it is the `assigns` (or `modifies`) clause: `__ESBMC_assigns(x)` says
"this call may change `x` and nothing else." The "nothing else" half is what
makes contracts useful — it lets a caller keep everything it already knew about
the rest of the state across the call, and it is exactly what `replace` mode
relies on when it havocs *only* the listed locations. The need to state this
explicitly (rather than leaving "what does not change" implicit) is the *frame
problem* in procedure specifications [1].

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
the [Loop Invariants](/docs/loop-invariants) guide. The "everything else is
unchanged" guarantee a frame condition provides is the contract-level analogue
of the *frame rule* in separation logic [2].

## See also

- [Function Contracts how-to guide](/docs/function-contracts) — syntax,
  `--enforce-contract` / `--replace-call-with-contract`, quantifiers,
  `__ESBMC_is_fresh`, and known limitations.
- [Loop Invariants](/docs/loop-invariants) — loop contracts and the frame rule.

## References

[1] Alexander Borgida, John Mylopoulos, Raymond Reiter: *On the Frame Problem in
Procedure Specifications.* IEEE Trans. Software Eng. 21(10):785–798, 1995.
[doi:10.1109/32.469460](https://doi.org/10.1109/32.469460)

[2] John C. Reynolds: *Separation Logic: A Logic for Shared Mutable Data
Structures.* LICS 2002: 55–74.
[doi:10.1109/LICS.2002.1029817](https://doi.org/10.1109/LICS.2002.1029817)
