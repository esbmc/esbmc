---
title: Python
weight: 120
next: /docs/python/overview
---

ESBMC-Python is a frontend for [ESBMC](https://github.com/esbmc/esbmc) that enables formal verification of Python programs. It converts Python source code into ESBMC's internal representation, allowing the engine to apply bounded model checking and SMT-based reasoning to detect bugs, assertion violations, and undefined behavior.

## In This Section

- [Overview](./overview) — Architecture, pipeline stages, and how the frontend works
- [Usage](./usage) — How to install prerequisites, invoke ESBMC on Python files, and interpret results
- [Supported Features](./supported-features) — Full reference of supported language constructs, data structures, and standard library modules
- [Limitations](./limitations) — Known restrictions in the current version
- [Pytest Test Generation](./PYTEST_TESTGEN) — Automatic generation of pytest test cases from counterexamples
- [Random Operational Model](./Random-Operational-Model-in-ESBMC) — Details on the `random` module stub
