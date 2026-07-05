---
title: Operational Models
---

Operational models (OMs) are ESBMC's hand-written models of C/C++ library and
runtime behaviour, compiled to GOTO so the verifier can reason about programs
that call them. These guides cover writing, fixing, and tracking OMs.

{{< cards >}}
  {{< card link="/docs/development/om/internal-c-and-cpp-operational-models" title="Internal C and C++ Operational Models" subtitle="How ESBMC's internal C and C++ operational models work." >}}
  {{< card link="/docs/development/om/guidelines" title="Fixing Guidelines" subtitle="Diagnosing and fixing operational-model bugs." >}}
  {{< card link="/docs/development/om/om-workload-estimate-and-tracking" title="Workload Estimate and Tracking" subtitle="Tracking the operational-model implementation effort." >}}
{{< /cards >}}
