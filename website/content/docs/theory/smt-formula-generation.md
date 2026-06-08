---
title: SMT Formula Generation
weight: 10
---

## Overview

ESBMC works by encoding a program's safety properties as an SMT formula and
passing it to an SMT solver. By default, solving happens internally and only the
verification verdict is reported. You can instead ask ESBMC to write the formula
to a file, which is useful for:

- Inspecting how ESBMC encodes a particular program or property
- Feeding the formula to an external SMT solver or analysis tool
- Debugging unexpected verification results

## Generating a Formula

Use `--smtlib --smt-formula-only --output` to dump the formula in
[SMT-LIB v2](https://smtlib.cs.uiowa.edu/) format without running the solver:

```sh
esbmc file.c --smtlib --smt-formula-only --output formula.smt2
```

`--smtlib` selects the SMT-LIB text backend; `--smt-formula-only` stops ESBMC
after encoding, skipping the solve step.

### Simple programs

ESBMC's simplifier may pre-resolve trivially safe programs at compile time,
producing an empty formula. Add `--unwind 1 --no-simplify` to force full
symbolic encoding:

```sh
esbmc file.c --smtlib --smt-formula-only --output formula.smt2 \
  --unwind 1 --no-simplify
```

Choose `--unwind N` to match the depth of loops in your program; for programs
without loops `--unwind 1` is sufficient.

## Solver-Specific Formats

Replace `--smtlib` with a solver flag to get the formula in a format native to
that backend:

```sh
# Z3 native format
esbmc file.c --z3 --smt-formula-only --output formula.smt2

# Bitwuzla native format
esbmc file.c --bitwuzla --smt-formula-only --output formula.smt2

# Similarly: --boolector, --yices, --mathsat, --cvc4
```

The `--smtlib` output is the most portable option — it follows the SMT-LIB v2
standard and is accepted by all compliant solvers, including CVC5, even though
ESBMC does not yet have a built-in CVC5 backend.

## About Formula Size

The generated formula is typically larger than you might expect from the source
code alone. ESBMC models the full memory address space and the lifetimes of
dynamically allocated objects, so a significant portion of the formula captures
pointer-safety bookkeeping rather than program logic.

## Supported SMT Theories

ESBMC selects the SMT theory based on the program and the active solver backend:

| Theory       | Description                                        | Typical use              |
|--------------|----------------------------------------------------|--------------------------|
| QF_BV        | Quantifier-free bit-vectors                        | Integer / pointer safety |
| QF_BVFP      | Bit-vectors + floating-point (IEEE 754)            | `--floatbv` mode         |
| QF_ALIA      | Arrays + linear integer arithmetic                 | Array safety checks      |
| QF_AUFLIRA   | Arrays + uninterpreted functions + mixed arithmetic| General C programs       |

Use `--smtlib` to see exactly which logic declaration (`(set-logic ...)`) ESBMC
emits for your program.
