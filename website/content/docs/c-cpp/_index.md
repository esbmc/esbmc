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
`--unwind`. The same holds for the `__builtin_` spellings of `<math.h>`
functions (`__builtin_frexp` writes its exponent), and `__builtin_abs`,
`__builtin_labs`, `__builtin_llabs`, `__builtin_popcountg`, `__builtin_clrsb`,
the `__builtin_rotateleft*`/`rotateright*` and `__builtin_bitreverse*` families
are computed exactly. The `__atomic_<op>_fetch`, `__sync_<op>_and_fetch`,
`__sync_bool_compare_and_swap`, `__sync_val_compare_and_swap`,
`__sync_lock_test_and_set`, `__sync_lock_release` and `__atomic_exchange`
builtins update the object atomically and return the value C11 and GCC specify,
where they previously left it unchanged.

These C library functions gained operational models, so a call reads and
writes memory and returns the value the standard specifies rather than a
nondeterministic one:

- `qsort` (an insertion sort that swaps whole elements) and `bsearch`. A
  comparator whose parameter types do not match the call is reported as
  undefined behaviour (C11 6.3.2.3p8)
- `strnlen`, `strndup` (copies at most `n` bytes into a `malloc`ed buffer, so
  an over-read or a leaked copy is caught), `strcasecmp`, `strncasecmp`
- `log10` and `cbrt`/`cbrtf` from musl, `cbrtl`, and `ilogb`/`logb` with their
  `f` and `l` forms; `atof` as `strtod(nptr, NULL)`
- `pthread_once`, which C11 `call_once` delegates to: one caller runs the
  routine and the others wait for it to finish

Several existing models were corrected: `strtol` and its siblings parse digits
by base and set `endptr` and `errno` as C11 7.22.1.4 requires; `strtod` rounds
correctly when the digits and decimal exponent are exact; `atan2`, `hypot`,
`ldexp`/`scalbn`, `nextafter` and `fesetround` follow C11 for signed zeros,
infinities, NaNs, subnormals, overflow and unrecognised modes; and `frexpl`,
`ldexpl` and their relatives read `long double` in the binary128 layout ESBMC
encodes.

`setjmp` and `longjmp` transfer control: `setjmp` returns 0 when called
directly, and a later `longjmp(env, v)` resumes after the matching `setjmp`,
which then returns `v` (or 1 if `v` is 0), returning through the frames in
between; code after the `longjmp` is unreachable. The lowering is skipped under
`--enable-unreachability-intrinsic`
([#8312](https://github.com/esbmc/esbmc/pull/8312)).

When `main` returns, the handlers registered with `atexit` run, and in C++ the
destructors of objects with static storage duration run after them —
constructed function-local statics first, then globals in reverse definition
order, skipping namespace `std`. An assertion in either is therefore checked,
and a global that frees memory in its destructor is not reported as a leak
([#8339](https://github.com/esbmc/esbmc/pull/8339)).

{{< cards >}}
  {{< card link="/docs/c-cpp/supported-features" title="C++ Support" subtitle="Which C++ language and STL features ESBMC supports." >}}
  {{< card link="/docs/c-cpp/limitations" title="C++ Limitations" subtitle="What ESBMC's C++ frontend does not yet handle, and the workarounds." >}}
  {{< card link="/docs/c-cpp/esbmc-cpp-workflow-and-resources" title="C++ Workflow and Resources" subtitle="Maintainer workflow and benchmark tracking for the C++ frontend." >}}
  {{< card link="/docs/c-cpp/ctest-gen" title="CTest Test Generation" subtitle="Materialise reached witnesses as runnable CTest cases." >}}
  {{< card link="/docs/c-cpp/html-reports" title="HTML Report Generation" subtitle="Generate browsable HTML reports of verification results." >}}
  {{< card link="/docs/c-cpp/reducing-c-programs" title="Reducing C Programs" subtitle="Shrink a failing C program to a minimal reproducer." >}}
{{< /cards >}}
