---
title: SMT
---

ESBMC encodes verification conditions into SMT and discharges them with backends
such as Bitwuzla, Z3, Boolector, CVC4/CVC5, Yices, and MathSAT. These guides
cover extending that backend.

{{< cards >}}
  {{< card link="/docs/development/smt/implement-a-new-smt-theory-into-esbmc" title="Implementing an SMT Theory" subtitle="Add a new SMT theory to ESBMC's encoding." >}}
  {{< card link="/docs/development/smt/integrate-a-new-smt-solver-into-the-esbmc-backend" title="Integrating a New SMT Solver" subtitle="Wire a new solver into the ESBMC backend." >}}
  {{< card link="/docs/development/smt/update-an-smt-solver-in-esbmc" title="Updating an SMT Solver" subtitle="Update an existing solver integration to a new version." >}}
{{< /cards >}}
