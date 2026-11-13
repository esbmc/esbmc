---
title: Lean Translation (Experimental)
weight: 13
---

`--goto2lean` prints the GOTO program that ESBMC builds from your source file as
Lean 4 definitions, together with a theorem stating that no ESBMC assertion
fails. Lean can then prove or refute that theorem, with automation or with a
proof written by you or by an LLM, and its kernel checks the result.

The translation works on the GOTO program, so it serves every front end that
produces one; it has been tested with C, C++ and Python.

{{< callout type="warning" >}}
This is a prototype. It covers a small fragment of programs (see
[What is supported](#what-is-supported)) and rejects everything else with an
error rather than translating it approximately.
{{< /callout >}}

## Getting started

You need [Lean 4](https://lean-lang.org/) (tested with 4.35). The generated
file imports only Lean core, so no Lake project or Mathlib is required.

```c
#include <assert.h>
unsigned nondet_uint(void);

unsigned mid(unsigned lo, unsigned hi) { return (lo + hi) / 2; }

int main(void) {
  unsigned lo = nondet_uint(), hi = nondet_uint();
  __ESBMC_assume(lo <= hi);
  unsigned m = mid(lo, hi);
  assert(lo <= m && m <= hi);
  return 0;
}
```

```bash
esbmc mid.c --goto2lean --output mid.lean
lean mid.lean
```

Lean reports the same counterexample as ESBMC (`lo + hi` wraps around):

```
mid.lean:30:12: error: The prover found a counterexample, consider the following assignment:
hi✝ = 4294967295#32
lo✝ = 4294967295#32
```

(The `✝` marks names that Lean introduced itself.)

With `return lo + (hi - lo) / 2;` instead, `lean mid.lean` prints nothing: the
theorem is proved.

Without `--output`, the translation goes to standard output. `--function name`
translates from another entry function; its parameters become universally
quantified in the theorem.

## What the translation means

Each function `f(args)` becomes `def f args (post : R → Prop) : Prop`: the
weakest precondition under which every assertion in `f` holds and `post` holds
of its result.

| GOTO instruction | Lean |
|---|---|
| nondeterministic value, uninitialised declaration | `∀ x : T, …` |
| `ASSUME c` | `c → …` |
| `ASSERT c` | `c ∧ …` |
| assignment | `let x := e; …` |
| call | `f args (fun r => …)` |
| branch | `(c → …) ∧ (¬c → …)`, with a shared join point |
| loop | a separate `coinductive_fixpoint` definition |

The generated theorem, `main_safe : Prog.main (fun _ => True)`, states
**partial correctness**: no assertion fails on any terminating execution.
Loops are greatest fixpoints, so the theorem says nothing about termination.

The assertions include ESBMC's own checks, which it inserts into the GOTO
program: array bounds, division by zero, and arithmetic overflow when you pass
`--overflow-check` or `--unsigned-overflow-check`. Fixed-width integers become
Lean `BitVec`s with C's wraparound, and `/` and `%` follow ESBMC's SMT encoding,
including the result for a zero divisor.

## Proving the theorem

For programs without loops, whose only quantifiers are asserted `∀`s, the
generated proof is automatic:
it unfolds the definitions and calls `bv_decide`, which either proves the
theorem or prints a counterexample.

For a program with a loop, the theorem ends in `sorry`: the proof needs a loop
invariant. Each loop definition comes with a coinduction principle; apply it
with the invariant and let `bv_decide` discharge the rest:

```lean
theorem main_safe : Prog.main (fun _ => True) := by
  intro n
  simp only []
  apply Prog.main_loop1.coinduct n _ _ (fun s i => s = i ∧ i ≤ n)
  · intro s i h
    simp only [and_true] at *
    bv_decide
  · bv_decide
```

A wrong invariant is refuted with a concrete counterexample, which makes this a
natural place for an LLM to propose an invariant and for Lean to check it.
An asserted `∀` (`__ESBMC_forall`), alone or in a conjunction, is proved
automatically: the script introduces the bound variable and `bv_decide` decides
the rest. An `∃` (`__ESBMC_exists`) needs a witness, and a quantifier in an
assumption, a branch condition or under a negation needs an instantiation, so
those goals also end in `sorry`.

## Unbounded integers

With `--ir`, integers become Lean's unbounded `Int`, matching ESBMC's
integer encoding, and the generated proof uses `omega` instead of `bv_decide`.
This is the mode to use for Python, whose integers do not overflow:

```bash
esbmc mid.py --function check --ir --goto2lean --output mid.lean
```

In this mode bitwise operators, shifts and overflow checks are rejected, and
so is division by a variable when `--no-div-by-zero-check` is given.

## What is supported

- `_Bool` and fixed-width integers, assignments, structured `if`/`else`,
  loops (without nested loops or calls inside them), early returns and
  non-recursive calls.
- Quantifiers in assertions and assumptions. Only an asserted `∀` is proved
  automatically; the rest end in `sorry`.
- Calls to functions without a body: their result becomes nondeterministic.

Rejected with `ERROR: goto2lean: unsupported …`: arrays, pointers, structs,
floating point, global variables, recursion, `printf`, ESBMC intrinsics,
unstructured jumps, and assertions in code that runs before the entry function
(C++ static constructors, Python module-level code). In Python, put the logic
in a function, select it with `--function`, and give the function a return
type other than `None`.

## Exit status

| Status | Meaning |
|---|---|
| 0 | The Lean file was written. |
| 1 | The program is outside the supported fragment, or the output file could not be written. |
| 6 | ESBMC failed before translating (for example, a parse error). |

On any failure, an earlier file at the `--output` path is removed, so a stale
proof is never left behind.
