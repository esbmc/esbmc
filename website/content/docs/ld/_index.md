---
title: Ladder Diagram
weight: 130
---

SAFE-LD is ESBMC's frontend for **IEC 61131-3 Ladder Diagram (LD)** programs —
the graphical language used to program Programmable Logic Controllers (PLCs).
It turns a Ladder Diagram, exported as PLCopen XML, into ESBMC's internal
representation so the engine's bounded model checker and SMT backend can verify
safety properties of the control logic — or prove their absence.

Properties are not written in the LD program itself; you supply them separately
in a small YAML file. The frontend models the PLC's cyclic scan (read inputs →
evaluate rungs → drive outputs, repeated forever) and checks each property on
every scan iteration.

> The frontend is gated behind the `ENABLE_LD_FRONTEND` build option (off by
> default). See [Usage](/docs/ld/usage) for how to build and invoke it.

{{< cards >}}
  {{< card link="/docs/ld/overview" title="Overview" subtitle="The cyclic-scan model and the pipeline from PLCopen XML to SMT formula." >}}
  {{< card link="/docs/ld/usage" title="Usage" subtitle="Build with the LD frontend, run esbmc on a .ld program, and read the result." >}}
  {{< card link="/docs/ld/property-format" title="Property Format" subtitle="The YAML specification for the five supported property kinds." >}}
  {{< card link="/docs/ld/limitations" title="Limitations" subtitle="Supported constructs and current restrictions." >}}
{{< /cards >}}
