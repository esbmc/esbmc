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

ESBMC uses a modular algorithm system for processing SSA (Static Single Assignment) steps during verification. The core architecture is built around the `ssa_step_algorithm` base class defined in `src/util/algorithms.h`.

## Algorithm Architecture

The SSA algorithm system follows these key design patterns:

1. **Base Interface**: All SSA algorithms inherit from `ssa_step_algorithm` which provides:
   - Virtual methods for different SSA step types (`run_on_assignment`, `run_on_assert`, etc.).
   - An `ignored()` method to track how many steps were optimized away.
   - A `run()` method that processes SSA step collections.

2. **Algorithm Registration**: In the BMC constructor, algorithms are initialized based on command-line options:
   - `assertion_cache` for caching assertions (when not using k-induction or forward conditions).
   - `symex_slicet` or `simple_slice` for program slicing.

## Key SSA Algorithms

### 1. Assertion Caching

The `assertion_cache` algorithm stores verification results between runs to avoid redundant checks.

### 2. Program Slicing

Two slicing algorithms are available:

- `symex_slicet`: Advanced slicing with dependency analysis.
- `simple_slice`: A no-op placeholder used when slicing is disabled via `--no-slice`, allowing the pipeline to run without performing any slice optimizations.

### 3. SSA Feature Analysis

The `ssa_features` class analyzes SSA programs to detect specific features like non-linear operations, bitwise operations, arrays, and structs. This is **not** part of the standard verification pipeline — it is only loaded when the `--ssa-features-dump` flag is passed.

## Algorithm Execution

During verification, algorithms run on each SSA step in sequence via the `run_thread` method.

# SMT Algorithms

TODO

