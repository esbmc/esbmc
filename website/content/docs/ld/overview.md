---
title: Overview
weight: 1
---

SAFE-LD verifies IEC 61131-3 Ladder Diagram programs by modelling the PLC's
cyclic scan and encoding user-supplied safety properties as assertions checked
on every scan iteration.

## The cyclic-scan model

A PLC executes its program in an endless loop called the *scan cycle*:

1. **Read inputs** — sample the physical input variables.
2. **Evaluate rungs** — compute coil and function-block outputs from the current
   inputs and retained state.
3. **Write outputs** — drive the physical outputs.

The frontend reproduces this faithfully. Each scan re-samples every input
variable non-deterministically, so verification explores *all* possible input
sequences rather than a single fixed trace. The rung logic and the retained
state of timers and counters are evaluated exactly as a PLC would, and the whole
body sits inside a `while(true)` scan loop. Properties are appended to the scan
body, so they are checked once per cycle, for every reachable state.

## Pipeline

```
PLCopen .xml  ──►  parse  ──►  LD IR  ──►  GOTO program  ──►  SSA  ──►  SMT
   (.ld)                     (rungs,      (scan loop +
                              FBs, vars)   assertions)
```

- **Parse** — the PLCopen XML is read into an internal Ladder Diagram IR of
  rungs, contacts, coils, and function-block instances.
- **IR generation** — the IR is lowered to a GOTO program: a scan loop whose
  body reads inputs, evaluates each rung, and updates timer/counter state.
- **Property encoding** — each entry in the YAML property file is turned into an
  assertion (`code_assertt`) spliced into the scan body. The property's
  `description` is carried into the assertion comment, so a violated property is
  named in the counterexample.
- **Backend** — from there it is the standard ESBMC pipeline: symbolic execution
  to SSA, SMT encoding, and solving.

## Verification strategies

Because the scan loop is unbounded, the strategy you choose determines what a
result means:

- **k-induction** (`--k-induction --unlimited-k-steps`) — attempts an unbounded
  proof. A successful run proves the property holds on *every* scan cycle. This
  is the strategy to use when you want to establish safety.
- **Bounded model checking** (`--incremental-bmc`) — explores scan cycles up to
  a bound. It is complete for finding violations within that bound: a reported
  counterexample is a genuine bug. Absence of a counterexample only means "no
  violation up to the bound", not a full proof.

See [Usage](/docs/ld/usage) for concrete invocations and
[Property Format](/docs/ld/property-format) for the soundness and completeness
guarantees of each property kind.
