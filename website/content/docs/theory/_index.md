---
title: Theory
weight: 500
---

This section covers the theory and internals behind ESBMC: how it models
programs, the proof algorithms it runs, how verification conditions are encoded
for SMT solvers, and the data structures it is built on.

{{< cards >}}
  {{< card link="/docs/theory/non-determinism" title="Modeling with Non-determinism" subtitle="The primitives ESBMC adds to C for non-deterministic values, assumptions, and atomic blocks." >}}
  {{< card link="/docs/theory/verification-algorithms" title="Verification Algorithms" subtitle="Falsification, incremental BMC, and k-induction — how ESBMC unwinds loops and proves correctness." >}}
  {{< card link="/docs/theory/counterexample-guided-abstract-refinement" title="Counterexample-Guided Abstraction Refinement" subtitle="Integer vs. bit-vector encodings and how ESBMC refines the abstraction." >}}
  {{< card link="/docs/theory/smt-formula-generation" title="SMT Formula Generation" subtitle="How symbolic execution turns a program into SSA and then into an SMT formula." >}}
  {{< card link="/docs/theory/ltl" title="Linear Temporal Logic" subtitle="LTL property checking via Büchi automata over finite prefixes." >}}
  {{< card link="/docs/theory/irep2" title="IRep2" subtitle="ESBMC's typed, reference-counted, copy-on-write internal representation." >}}
{{< /cards >}}
