---
title: Python
weight: 120
next: /docs/python/overview
---

ESBMC-Python is a frontend for [ESBMC](https://github.com/esbmc/esbmc) that
enables formal verification of Python programs. It converts Python source code
into ESBMC's internal representation, allowing the engine to apply bounded model
checking and SMT-based reasoning to detect bugs, assertion violations, and
undefined behavior.

{{< cards >}}
  {{< card link="/docs/python/overview" title="Overview" subtitle="Architecture, pipeline stages, and how the frontend works." >}}
  {{< card link="/docs/python/usage" title="Usage" subtitle="Invoke ESBMC on Python files and interpret the results." >}}
  {{< card link="/docs/python/supported-features" title="Supported Features" subtitle="Supported language constructs, data structures, and standard library modules." >}}
  {{< card link="/docs/python/limitations" title="Limitations" subtitle="Known restrictions in the current version." >}}
  {{< card link="/docs/python/pytest-testgen" title="Pytest Test Generation" subtitle="Generate pytest test cases automatically from counterexamples." >}}
  {{< card link="/docs/python/random-operational-model" title="Random Operational Model" subtitle="How the random module stub is modelled." >}}
{{< /cards >}}
