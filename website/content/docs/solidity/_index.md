---
title: Solidity
weight: 110
---

ESBMC-Solidity is a frontend that turns Solidity smart contract sources into
ESBMC's internal representation, so the engine's bounded model checker and SMT
backend can verify program properties — assertion violations, integer overflow,
reentrancy, and similar bug classes — or prove their absence.

You don't write a `main`. The frontend auto-generates a dispatcher harness that
calls public functions in arbitrary order with arbitrary arguments, and ESBMC
explores every reachable state symbolically.

{{< cards >}}
  {{< card link="/docs/solidity/overview" title="Overview" subtitle="Pipeline from .sol source to SMT formula; what BMC and k-induction mean." >}}
  {{< card link="/docs/solidity/usage" title="Usage" subtitle="Prerequisites, flags, and worked examples." >}}
  {{< card link="/docs/solidity/supported-features" title="Supported Features" subtitle="Solidity language features the frontend recognises." >}}
  {{< card link="/docs/solidity/limitations" title="Limitations" subtitle="Known gaps and current solver caveats." >}}
{{< /cards >}}
