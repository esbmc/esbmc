---
title: Overview
weight: 1
---

The Solidity frontend converts smart contract source files into ESBMC's internal representation, allowing the engine's bounded model checker and SMT backend to verify program properties.

## What does ESBMC check?

ESBMC explores program executions symbolically — instead of running the contract on concrete inputs, it reasons about *all* inputs at once and asks an SMT solver whether a property can ever fail.

Two verification modes are relevant for Solidity:

- **Bounded model checking (BMC)** — Unrolls loops up to a fixed bound `N` (`--unwind N`) and checks every reachable state for property violations. If a counterexample exists within `N` iterations, ESBMC reports it. If no counterexample exists, the result is "no bug *up to bound N*" — soundness for bug-finding, but not a proof of full correctness beyond `N`.
- **k-induction** (`--k-induction`) — Combines a base-case BMC check with an inductive step that tries to prove the property holds for *every* iteration, regardless of bound. When successful, this yields a full proof of correctness.

You do not need to choose by hand: BMC is the default, and `--k-induction` opts into the inductive proof rule. See [Usage](./usage) for examples of each.

## Pipeline

The frontend has four stages before ESBMC's backend takes over:

```
contract.sol
   │
   ▼  (1) solc --ast-compact-json
contract.solast (JSON AST)
   │
   ▼  (2) AST normalisation & multi-file merge
clean AST
   │
   ▼  (3) AST → IRep2 (type/expr/decl converters)
symbol table
   │
   ▼  (4) harness generation
GOTO program  ──►  symbolic execution  ──►  SMT formula  ──►  solver
```

### 1. Solc Invocation

ESBMC invokes the [Solidity compiler](https://soliditylang.org) to obtain a JSON Abstract Syntax Tree:

```bash
solc --ast-compact-json contract.sol
```

The solc binary is resolved in this order: `--solc-bin <path>`, then the `$SOLC` environment variable, then `solc` on `$PATH`. Solidity ≥ 0.5.0 is supported; ≥ 0.7 is recommended.

You can also bypass solc entirely by supplying a pre-generated AST as `contract.solast`:

```bash
solc --ast-compact-json contract.sol > contract.solast
esbmc --sol contract.sol contract.solast --contract MyContract
```

### 2. AST Normalisation

The frontend strips JSON-null fields (which different solc versions emit inconsistently) and merges multi-file imports into a single AST.

### 3. AST → IRep2

The cleaned AST is traversed and each contract, function, type, and expression is mapped to ESBMC's typed internal representation (IRep2). State variables become module-level symbols; functions become first-class methods; `mapping`, dynamic arrays, and structs lower to operational models in `src/c2goto/library/solidity/`.

### 4. Harness Generation

Smart contract verification differs from C verification: there is no `main()`. ESBMC synthesises one automatically — a *dispatcher harness*:

```c
constructor();                     // run state initialisation
while (nondet_bool()) {            // unbounded sequence of calls
    set_msg_context();             // nondet sender, value, etc.
    switch (nondet_uint()) {       // nondet function selection
        case 0: f1(...); break;
        case 1: f2(...); break;
        ...
    }
    // assertions are checked after every call
}
```

This explores every reachable contract state across every possible sequence of public-function calls. Use `--focus-function <name>` to restrict the dispatcher to a single function while still running the constructor.

### 5. Backend

Once the GOTO program is built, ESBMC's backend runs symbolic execution to produce SSA (static single-assignment) form, encodes the result as a first-order SMT formula, and discharges it with an SMT solver.

The default solver for Solidity is **Bitwuzla**. Pass `--bitwuzla`, `--z3`, or `--cvc5` to pick a specific backend.
