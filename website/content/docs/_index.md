---
title: Documentation
next: /docs/setup
---

ESBMC is an SMT-based bounded model checker for C, C++, CUDA, CHERI, Python,
Solidity, Java and Kotlin. It detects — or proves the absence of — runtime
errors and verifies user-defined assertions, without requiring pre- or
post-conditions.

New here? Start with [Setup](/docs/setup), then the
[Build Guide](/docs/development/building) and [Usage](/docs/usage).

## What ESBMC checks

By default, ESBMC checks for:

- User-specified assertion failures
- Out-of-bounds array access
- Illegal pointer dereferences (null, out-of-bounds, double-free, misalignment)
- Integer overflow
- Floating-point NaN
- Division by zero
- Memory leaks

For concurrent (pthread) programs it can additionally check for deadlock, data
races, atomicity violations and lock-ordering issues by explicitly exploring
thread interleavings. Each reported violation is annotated with its matching
[CWE identifier](/docs/cwe-mapping).

## Explore the docs

{{< cards >}} {{< card link="/docs/setup" title="Setup" >}}
{{< card link="/docs/development/building" title="Build Guide" >}}
{{< card link="/docs/usage" title="Usage" >}}
{{< card link="/docs/constructs" title="Constructs" >}}
{{< card link="/docs/function-contracts" title="Function Contracts" >}}
{{< card link="/docs/loop-invariants" title="Loop Invariants" >}}
{{< card link="/docs/coverage" title="Coverage" >}}
{{< card link="/docs/c-cpp" title="C / C++" >}}
{{< card link="/docs/python" title="Python" >}}
{{< card link="/docs/solidity" title="Solidity" >}}
{{< card link="/docs/cwe-mapping" title="CWE Mapping" >}}
{{< card link="/docs/development" title="Development" >}} {{< /cards >}}

## Background theory

To understand how ESBMC reasons about programs, read the background theory:

{{< cards >}}
{{< card link="/docs/theory/non-determinism" title="Modeling with Non-determinism" >}}
{{< card link="/docs/theory/verification-algorithms" title="Verification Algorithms" >}}
{{< card link="/docs/theory" title="More Theory" >}} {{< /cards >}}

## Further reading

- Slides on detecting software vulnerabilities with ESBMC:
  [Part I](https://ssvlab.github.io/lucasccordeiro/courses/2022/01/software-security/slides/lecture03.pdf),
  [Part II](https://ssvlab.github.io/lucasccordeiro/courses/2022/01/software-security/slides/lecture04.pdf),
  [Part III](https://ssvlab.github.io/lucasccordeiro/courses/2022/01/software-security/slides/lecture05.pdf)
- The
  [software security](https://ssvlab.github.io/lucasccordeiro/courses/2022/01/software-security/index.html)
  course covers further implementation details about ESBMC.

## Support

ESBMC is under active development, with new features, optimizations and
encodings added continuously. We invite our users to submit bug reports to the
[GitHub repository](https://github.com/esbmc/esbmc/issues). For any question,
reach us via [GitHub Discussions](https://github.com/esbmc/esbmc/discussions) or
our public ESBMC
[Zulip Channel](https://systemsandsoftwaresecurity.zulipchat.com/join/6ajabxaurkcwl6tl3ify55jv/).
