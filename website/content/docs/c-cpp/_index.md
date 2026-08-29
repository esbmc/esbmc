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

## C language and library support

Beyond standard C, the frontend handles the GNU extensions that appear in real
code — statement expressions, `__attribute__`, `typeof`, computed `goto`
(`goto *label_ptr` over a `void *` label array), and `__label__` local label
declarations.

On the library side, C11 concurrency is modelled as well as POSIX threads:
`<threads.h>` (`thrd_*`, `mtx_*`, `cnd_*`, `tss_*`) has an operational model
lowered onto the same pthread model, and the `__c11_atomic_*` builtins behind
`<stdatomic.h>` are modelled directly. The pthread model itself covers barriers,
spinlocks, read/write locks, and the recursive and error-checking mutex kinds
selected through `pthread_mutexattr_settype` — see
[Concurrency](/docs/theory/concurrency).

Clang's `__builtin_` spellings of the memory and string routines — including
`__builtin_memset`, `__builtin_memcmp`, `__builtin_strncpy` and
`__builtin_calloc` — are rewritten to the names ESBMC models, so they behave as
the plain calls do instead of going nondeterministic. A rewritten call runs the
modelled loop, so a builtin applied to a large object needs an adequate
`--unwind`.

{{< cards >}}
  {{< card link="/docs/c-cpp/supported-features" title="C++ Support" subtitle="Which C++ language and STL features ESBMC supports." >}}
  {{< card link="/docs/c-cpp/limitations" title="C++ Limitations" subtitle="What ESBMC's C++ frontend does not yet handle, and the workarounds." >}}
  {{< card link="/docs/c-cpp/esbmc-cpp-workflow-and-resources" title="C++ Workflow and Resources" subtitle="Maintainer workflow and benchmark tracking for the C++ frontend." >}}
  {{< card link="/docs/c-cpp/ctest-gen" title="CTest Test Generation" subtitle="Materialise reached witnesses as runnable CTest cases." >}}
  {{< card link="/docs/c-cpp/html-reports" title="HTML Report Generation" subtitle="Generate browsable HTML reports of verification results." >}}
  {{< card link="/docs/c-cpp/reducing-c-programs" title="Reducing C Programs" subtitle="Shrink a failing C program to a minimal reproducer." >}}
{{< /cards >}}
