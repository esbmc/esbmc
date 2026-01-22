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
TODO

# SMT Algorithms
TODO
