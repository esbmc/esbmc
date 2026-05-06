---
title: Solidity
weight: 110
---

ESBMC-Solidity is a frontend that turns Solidity smart contract sources into ESBMC's internal representation, so the engine's bounded model checker and SMT backend can verify program properties — assertion violations, integer overflow, reentrancy, and similar bug classes — or prove their absence.

You don't write a `main`. The frontend auto-generates a dispatcher harness that calls public functions in arbitrary order with arbitrary arguments, and ESBMC explores every reachable state symbolically.

- [Overview](./overview) — pipeline from `.sol` source to SMT formula; what BMC and k-induction mean.
- [Usage](./usage) — prerequisites, flags, and worked examples.
- [Supported Features](./supported-features) — Solidity language features the frontend recognises.
- [Limitations](./limitations) — known gaps and current solver caveats.
