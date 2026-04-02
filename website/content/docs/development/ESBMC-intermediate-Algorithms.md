---
title: Intermediate Algorithms
---

In ESBMC, we have the following byproducts for each phase:

- GOTO program (after the frontend generation).
- SSA trace (after the symbolic execution).
- SMT formula (after translating the SSA trace).

This provides the tool with modularization opportunities.

# GOTO Programs

## Abstract Interpretation (`goto-programs/ai.h`)

- Interval Analysis
- Static Analysis (it is defined in the terms of AI).

## Static Analysis

- Value Set Analysis

## GOTO Functions Algorithm (`algorithms/algorithm.h`).

![classgoto__functions__algorithm](https://user-images.githubusercontent.com/8601807/195880520-f4d333c4-a0f9-4fca-8645-c3d329f23d09.png)

- Goto Contractor
- Unsound Loop Unroller
- Bounded Loop Unroller
- Mark Declarations as Non Deterministic

## Other GOTO algorithms

There are few algorithms that could be turned into explicit algorithms. However, right now they are independent functions:

- Add Race Assertions
- Goto Check
- Goto Inline
- Goto K-Induction
- Goto Loops
- Remove Skips
- Remove Unreachable

# SSA Algorithms

ESBMC uses a modular algorithm system for processing SSA (Static Single Assignment) steps during verification. The core architecture is built around the `ssa_step_algorithm` base class defined in `src/util/algorithms.h` (algorithms.h:86-119).

## Algorithm Architecture

The SSA algorithm system follows these key design patterns:

1. **Base Interface**: All SSA algorithms inherit from `ssa_step_algorithm` which provides:
   - Virtual methods for different SSA step types (`run_on_assignment`, `run_on_assert`, etc.).
   - An `ignored()` method to track how many steps were optimized away.
   - A `run()` method that processes SSA step collections (algorithms.h:86-119).

2. **Algorithm Registration**: In the BMC constructor, algorithms are initialized based on command-line options (bmc.cpp:61-80):
   - `assertion_cache` for caching assertions (when not using k-induction or forward conditions).
   - `symex_slicet` or `simple_slice` for program slicing.
   - `ssa_features` for feature analysis.

## Key SSA Algorithms

### 1. Assertion Caching

The `assertion_cache` algorithm stores verification results between runs to avoid redundant checks (bmc.cpp:64-71).

### 2. Program Slicing

Two slicing algorithms are available:

- `symex_slicet`: Advanced slicing with dependency analysis.
- `simple_slice`: Basic slicing when `--no-slice` is specified (bmc.cpp:73-76).

### 3. SSA Feature Analysis

The `ssa_features` class analyzes SSA programs to detect specific features like non-linear operations, bitwise operations, arrays, and structs (features.h:7-35).

## Algorithm Execution

During verification, algorithms run on each SSA step in sequence via the `run_thread` method (bmc.cpp:1187-1191).

# SMT Algorithms

ESBMC's SMT layer provides a unified interface to multiple SMT solvers through the `smt_convt` base class (smt_conv.h:133-199).

## SMT Architecture

The SMT system uses a three-layer abstraction:

1. **Base Classes**: `smt_convt`, `smt_ast`, and `smt_sort` provide the core interface (README.txt:12-18).
2. **Solver Implementations**: Each solver (Z3, Boolector, MathSAT, CVC4/CVC5, etc.) subclasses these base classes.
3. **Conversion Pipeline**: Expressions are flattened and converted through `mk_func_app` and related methods (smt_conv.h:24-31).

## Supported SMT Solvers

ESBMC supports multiple SMT solvers (README.md:48-57):

- Z3 4.13+
- Boolector 3.0+
- MathSAT
- CVC4/CVC5
- Yices 2.2+
- Bitwuzla

## SMT Conversion Process

The conversion from SSA to SMT happens in `symex_target_equationt::convert_internal_step` (symex_target_equation.cpp:166-248):

1. **Expression Conversion**: Each SSA step is converted to SMT AST.
2. **Guard Handling**: Step guards are converted to SMT expressions.
3. **Assertion Processing**: Assertions are converted with implication logic.
4. **Assumption Accumulation**: Assumptions are conjoined using `mk_and`.

## Encoding Modes

ESBMC supports two main encoding modes (bmc.cpp:194-201):

- **Bit-vector mode**: Uses precise bit-vector arithmetic.
- **Integer/Real mode**: Uses unbounded integers for better performance.

## Specialized SMT Features

### Floating-Point Handling

When `--ir-ra` is enabled, floating-point operations are encoded using real arithmetic with error enclosures (README.txt:52-83).

### Array and Tuple Support

The SMT layer includes virtual interfaces for arrays and tuples, allowing solvers to provide optimized implementations (smt_conv.h:120-127).

## Integration Pipeline

The SSA and SMT algorithms work together in ESBMC's verification pipeline:

1. **Symbolic Execution**: Generates SSA steps.
2. **SSA Processing**: Algorithms optimize and analyze the SSA program.
3. **SMT Conversion**: SSA steps are converted to SMT formulas.
4. **Solver Execution**: SMT solver determines satisfiability.
5. **Result Processing**: Counterexamples are generated if needed.

This modular design allows ESBMC to support multiple verification strategies and solver backends while maintaining a consistent verification pipeline.

## Notes

- The SSA algorithm system is designed to be extensible, with new algorithms easily added through the `ssa_step_algorithm` interface.
- SMT solver selection is handled through a factory pattern in `create_solver()` function.
- The slicing algorithms can significantly reduce verification time by removing irrelevant code paths.
- ESBMC's SMT conversion handles complex C constructs like pointers, structs, and unions through flattening techniques.
