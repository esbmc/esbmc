---
title: Verification Algorithms
weight: 20
---

ESBMC [1] implements several incremental verification strategies on top of its
symbolic execution engine. This page explains how each works; for the practical
flags and when to use them, see the [Usage](/docs/usage) guide.

## Bounded Model Checking {#bmc}

```sh
esbmc file.c --unwind K
```

Plain bounded model checking is the foundation the strategies below build on.
ESBMC symbolically executes the program, unrolling each loop at most *K* times,
encodes the resulting [SSA program and safety properties as a single SMT
formula](/docs/theory/smt-formula-generation) [2], and asks the solver whether
any property can be violated. A satisfiable formula yields a counterexample; an
unsatisfiable one means no property is violated *within the bound K*.

Because a fixed bound can cut a loop short, ESBMC adds an **unwinding assertion**
after each loop that fails if the loop needed more than *K* iterations — so an
incompletely unrolled loop is reported rather than silently assumed safe. This
makes a `VERIFICATION SUCCESSFUL` under `--unwind K` a *bounded* proof. The
checks can be relaxed with `--no-unwinding-assertions` (turn unwinding
assertions into assumptions) or `--partial-loops` (permit partial loop paths),
and per-loop bounds can be set with `--unwindset L:nr`. The strategies below
remove the need to pick *K* by hand by iterating the bound automatically.

## Falsification {#falsification}

```sh
esbmc file.c --falsification
```

The falsification approach uses an iterative technique and verifies the program
for each unwind bound up to a maximum default value of 50 (changeable via
`--max-k-step N`, or `--unlimited-k-steps` to run until it exhausts the time or
memory limit).
The goal is to find a counterexample with up to *k* loop unwindings; the
symbolic execution engine increasingly unwinds the loop after each iteration.

This approach replaces all unwinding assertions (e.g. assertions checking
whether a loop was completely unrolled) with unwinding assumptions. Normally
this would be unsound, but since falsification cannot provide a correctness
proof, it does not affect the search for bugs — it only cares whether the
current number of unwindings leads to a property violation.

Falsification can also change the granularity of the increment (default 1) via
`--k-step N`. A larger increment can slow verification and may not produce the
shortest counterexample.

## Incremental BMC {#incremental-bmc}

```sh
esbmc file.c --incremental-bmc
```

Incremental BMC also iterates over unwind bounds (default max 50, via
`--max-k-step N`, or `--unlimited-k-steps`). It either finds a counterexample
within *k* unwindings or fully unwinds all loops to provide a correctness
result.

The approach has two steps. When searching for a property violation it replaces
unwinding assertions with assumptions (reporting an unwinding-assertion failure
is not a real bug). It then checks whether all loops were fully unrolled, by
checking whether all unwinding assertions are unsatisfiable; the other
assertions need not be re-checked for the current *k*, as they were already
verified.

As with falsification, the increment granularity is configurable via
`--k-step N`.

## k-Induction proof rule {#k-induction}

```sh
esbmc file.c --k-induction
```

The original *k*-induction algorithm by Sheeran et al. [3] proved safety
properties in hardware verification; it was later refined by Donaldson et al.
[4] for general C programs. ESBMC combines both approaches, with an algorithm
tailored to handling loops in C [5]:

```
¬B(k)         → program contains a bug
B(k) ∧ F(k)   → program is correct
B(k) ∧ I(k)   → program is correct
```

Here *B(k)* is the base case, *F(k)* the forward condition and *I(k)* the
inductive step; *k* is the number of loop unwindings used for each step. The
base case uses plain BMC, so it can only find property violations; if its error
check is satisfiable, the algorithm reports a counterexample of length *k*. For
the forward condition and inductive step, the base case must be checked first
(a soundness requirement).

The forward condition proves that all loops were fully unrolled by adding
unwinding assertions after each loop (and only checks those, since program
assertions are already discharged by the base case for the current *k*). The
inductive step proves that if the property holds for *k* iterations it holds for
the next, by assigning non-deterministic values to all variables written in a
loop body, assuming *k-1* invariants and checking the invariant at the *k*th
iteration.

The algorithm starts at *k = 1* and increases it, incrementally analysing the
program until it finds a bug (base case satisfiable), proves correctness (base
case unsatisfiable and either the forward condition or inductive step
unsatisfiable), or exhausts the time or memory limit.

## References

[1] Mikhail Y. R. Gadelha, Felipe R. Monteiro, Jeremy Morse, Lucas C. Cordeiro,
Bernd Fischer, Denis A. Nicole: *ESBMC 5.0: an industrial-strength C model
checker.* ASE 2018: 888–891. [doi:10.1145/3238147.3240481](https://doi.org/10.1145/3238147.3240481)

[2] Lucas C. Cordeiro, Bernd Fischer, João Marques-Silva: *SMT-Based Bounded
Model Checking for Embedded ANSI-C Software.* IEEE Trans. Software Eng.
38(4):957–974, 2012. [doi:10.1109/TSE.2011.59](https://doi.org/10.1109/TSE.2011.59)

[3] Mary Sheeran, Satnam Singh, Gunnar Stålmarck: *Checking Safety Properties
Using Induction and a SAT-Solver.* FMCAD 2000: 108–125

[4] Alastair F. Donaldson, Leopold Haller, Daniel Kroening, Philipp Rümmer:
*Software Verification Using k-Induction.* SAS 2011: 351–368

[5] Mikhail Y. R. Gadelha, Hussama Ibrahim Ismail, Lucas C. Cordeiro: *Handling
loops in bounded model checking of C programs via k-induction.* Int. J. Softw.
Tools Technol. Transf. 19(1):97–114, 2017. [doi:10.1007/s10009-015-0407-9](https://doi.org/10.1007/s10009-015-0407-9)
