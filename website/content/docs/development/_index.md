---
title: Development
weight: 140
---

Guides for building, understanding, and contributing to ESBMC. If you're new,
start with the [Build Guide](/docs/development/building) and the
[Architecture](/docs/development/architecture) overview.

## Understanding ESBMC

{{< cards >}}
  {{< card link="/docs/development/building" title="Build Guide" subtitle="Build ESBMC from source on your platform." >}}
  {{< card link="/docs/development/architecture" title="Architecture" subtitle="How the parser, GOTO program, symbolic execution, and SMT backend fit together." >}}
  {{< card link="/docs/development/esbmc-intermediate-algorithms" title="Intermediate Algorithms" subtitle="The core algorithms ESBMC runs between the frontend and the solver." >}}
{{< /cards >}}

## Extending ESBMC

{{< cards >}}
  {{< card link="/docs/development/frontend" title="Creating a Frontend" subtitle="Add support for a new source language." >}}
  {{< card link="/docs/development/adding-new-expressions" title="Adding New Expressions" subtitle="Extend the IRep2 internal representation with new expressions." >}}
  {{< card link="/docs/development/om" title="Operational Models" subtitle="Model C/C++ library and runtime behaviour for verification." >}}
  {{< card link="/docs/development/smt" title="SMT" subtitle="Implement theories and integrate or update SMT solvers." >}}
{{< /cards >}}

## Processes & infrastructure

{{< cards >}}
  {{< card link="/docs/development/releases" title="Releases" subtitle="How to prepare and publish a new ESBMC release." >}}
  {{< card link="/docs/development/ppa-maintenance" title="PPA Maintenance" subtitle="Maintaining the Ubuntu PPA packages." >}}
  {{< card link="/docs/development/benchexec" title="Benchexec" subtitle="Running ESBMC under BenchExec for reproducible benchmarking." >}}
  {{< card link="/docs/development/sv-comp-and-test-comp-faq" title="SV-COMP and Test-COMP" subtitle="FAQ for the competition configurations." >}}
{{< /cards >}}
