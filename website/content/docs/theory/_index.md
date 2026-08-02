---
title: Theory
weight: 500
---

This section covers the theory and internals behind ESBMC: how it models
programs, the proof algorithms it runs, how verification conditions are encoded
for SMT solvers, and the data structures it is built on.

{{< cards >}}
  {{< card link="/docs/theory/non-determinism" title="Modeling with Non-determinism" subtitle="The primitives ESBMC adds to C for non-deterministic values, assumptions, and atomic blocks." >}}
  {{< card link="/docs/theory/verification-algorithms" title="Verification Algorithms" subtitle="Bounded model checking, falsification, incremental BMC, and k-induction — how ESBMC unwinds loops and proves correctness." >}}
  {{< card link="/docs/theory/termination" title="Termination and Non-termination" subtitle="Proving loops halt via ranking functions, refuting them via recurrent sets, and the reduction to a k-induction reachability check." >}}
  {{< card link="/docs/theory/memory-model" title="Memory Model and Pointer Safety" subtitle="How ESBMC models pointers as object + offset and derives bounds, NULL, use-after-free, and leak checks." >}}
  {{< card link="/docs/theory/concurrency" title="Concurrency and Context-Bounded MC" subtitle="Thread interleavings, context bounding, partial-order reduction, and data-race / deadlock detection." >}}
  {{< card link="/docs/theory/interval-analysis" title="Interval Analysis" subtitle="Abstract interpretation that infers variable ranges and injects them as assumptions before SMT solving." >}}
  {{< card link="/docs/theory/function-contracts" title="Function Contracts" subtitle="Modular assume-guarantee verification with requires / ensures / assigns and loop contracts." >}}
  {{< card link="/docs/theory/counterexample-guided-abstract-refinement" title="Integer vs. Bit-Vector Encoding" subtitle="The --ir integer encoding versus default bit-vectors, and a manual refinement workflow." >}}
  {{< card link="/docs/theory/smt-formula-generation" title="SMT Formula Generation" subtitle="How symbolic execution turns a program into SSA and then into an SMT formula." >}}
  {{< card link="/docs/theory/ltl" title="Linear Temporal Logic" subtitle="LTL property checking via Büchi automata over finite prefixes." >}}
  {{< card link="/docs/theory/irep2" title="IRep2" subtitle="ESBMC's typed, reference-counted, copy-on-write internal representation." >}}
{{< /cards >}}
