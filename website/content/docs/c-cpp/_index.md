---
title: C and C++
weight: 100
---

C and C++ are ESBMC's primary frontends, built on Clang/LLVM. ESBMC verifies C
(up to C23) and C++ (up to C++20, with selected C++23 features), checking memory
safety, arithmetic overflow, pointer safety, concurrency properties, and user
assertions — or proving their absence.

To get started, see the [Usage](/docs/usage) guide and the
[Constructs](/docs/constructs) reference for the verification annotations. The
pages below cover C/C++-specific tooling and support.

{{< cards >}}
  {{< card link="/docs/c-cpp/esbmc-cpp-support" title="C++ Support" subtitle="Which C++ language and STL features ESBMC supports." >}}
  {{< card link="/docs/c-cpp/esbmc-cpp-workflow-and-resources" title="C++ Workflow and Resources" subtitle="A practical workflow and resources for verifying C++ programs." >}}
  {{< card link="/docs/c-cpp/ctest-gen" title="CTest Test Generation" subtitle="Materialise reached witnesses as runnable CTest cases." >}}
  {{< card link="/docs/c-cpp/html-reports" title="HTML Report Generation" subtitle="Generate browsable HTML reports of verification results." >}}
  {{< card link="/docs/c-cpp/reducing-c-programs" title="Reducing C Programs" subtitle="Shrink a failing C program to a minimal reproducer." >}}
{{< /cards >}}
