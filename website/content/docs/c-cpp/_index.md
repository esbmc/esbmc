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

`==` and `!=` on `_Complex` operands compare the real and imaginary parts as
real numbers, so `-0.0` equals `+0.0` and a NaN part compares unequal. A real
operand of complex `+`, `-`, `*` or `/` applies to each part, and `*` and `/`
recover the infinities that C17 G.5.1p4 requires, where the textbook formulas
give NaN. A `sizeof` operand of variable length array type is evaluated (C17
6.5.3.4p2), so for `int a[n][n]`, `sizeof(a[i++])` increments `i`. Under `--volatile-check`, every
read of a volatile object, including a struct member and a read through a
volatile-qualified pointer, returns a fresh nondeterministic value.

Under `--overflow-check`, a signed left shift with a non-negative left operand
overflows in C when the result does not fit the signed type (C11 6.5.7p4), so
`1 << 31` on a 32-bit `int` is reported. In C++ before C++20 it overflows only
when the result does not fit the corresponding unsigned type (CWG 1457), so
`1 << 31` is defined and `2 << 31` is reported; from C++20 every such shift is
defined.

See [Arguments of `main`](/docs/usage#arguments-of-main) for how `argv` and
`envp` are modelled.

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
- `strcasecmp_l`, `strncasecmp_l` and the `<ctype.h>` functions that take a
  `locale_t` (`isalpha_l` to `isxdigit_l`, `tolower_l`, `toupper_l`). Each
  behaves as its locale-free counterpart: only the C locale is modelled
- `nextafterl`, `nexttoward`, `nexttowardf` and `nexttowardl`
- `sem_trywait`, which takes an available semaphore or fails with `EAGAIN`
  without blocking, and `sem_timedwait`, which takes the semaphore, waits as
  `sem_wait` does, or fails with `ETIMEDOUT`. The timeout value is not modelled

Several existing models were corrected: `strtol` and its siblings parse digits
by base and set `endptr` and `errno` as C11 7.22.1.4 requires; `strtod` rounds
correctly when the digits and decimal exponent are exact; `atan2`, `hypot`,
`ldexp`/`scalbn`, `nextafter` and `fesetround` follow C11 for signed zeros,
infinities, NaNs, subnormals, overflow and unrecognised modes; and `frexpl`,
`ldexpl` and their relatives read `long double` in the binary128 layout ESBMC
encodes.

Other models now keep error and end-of-input paths reachable and check the
memory they touch:

- `calloc` returns `NULL` when `nmemb * size` wraps `size_t`, instead of
  pruning the path
- `posix_memalign` returns `EINVAL` only for an alignment that is not a power
  of two multiple of `sizeof(void *)` and places no condition on the size, so
  `posix_memalign(&p, 16, 24)` can succeed
- `fgets` returns `NULL` when end-of-file comes before any character, and
  `read` may return 0
- `getopt` returns a nondeterministic option, so every `case` of the usual
  `while ((c = getopt(...)) != -1) switch (c)` loop is explored
- `asctime`, `ctime` and their `_r` forms return a 25-character string in the
  C17 7.27.3.1 layout
- `fwrite` and `fputs` read their buffer, and `memcmp` checks that all `n`
  bytes of both objects can be read, so an over-read or a use-after-free
  through them is reported

ESBMC interprets the `printf` family itself. The value a call returns and the
characters `sprintf` and `snprintf` store follow C11 7.21.6.1: `%n` writes the
count so far through its pointer, a `*` width or precision reads its `int`
argument, `%e` and `%g` of a constant `double` use a two-digit exponent and the
`%g` style rule, `%ls` of an ASCII wide string literal prints one byte per
character, and integer and `%s` conversions store their exact characters even
for a nondeterministic argument (up to 64 output characters). `vsprintf` and
`vsnprintf` format their `va_list` arguments only on targets whose `va_list` is
a `char *`, such as macOS on Apple silicon.

By default ESBMC checks that a non-literal format string and each `%s`
argument without a precision or length modifier can be read, so `printf(p)`
after `free(p)` or a `%s` of an expired stack buffer is reported as a
dereference failure. `--printf-check` adds stricter checks: every pointer
argument is dereferenced, `%s` and `%p` must receive a pointer, the format may
not name more conversions than there are arguments, and a `%s` argument without
a precision or length modifier must hold a NUL within its object:

```c
#include <stdio.h>
int main(void)
{
  char tag[3] = {'a', 'b', 'c'};
  printf("%s\n", tag);
  return 0;
}
```

```sh
esbmc tag.c                  # VERIFICATION SUCCESSFUL
esbmc tag.c --printf-check   # VERIFICATION FAILED
                             # printf %s argument is not NUL-terminated within its object
```

The NUL check applies when the argument points at a constant offset into one
byte array with at most 1024 bytes from there to its end. The `--printf-check`
checks skip the `va_list` of `vprintf` and the other `v*` calls.

`setjmp` and `longjmp` transfer control: `setjmp` returns 0 when called
directly, and a later `longjmp(env, v)` resumes after the matching `setjmp`,
which then returns `v` (or 1 if `v` is 0), returning through the frames in
between; code after the `longjmp` is unreachable
([#8312](https://github.com/esbmc/esbmc/pull/8312)). `sigsetjmp`,
`siglongjmp`, `_setjmp` and `_longjmp` are lowered the same way, as is a
`longjmp` called through a function pointer, and the lowering also runs under
`--enable-unreachability-intrinsic`. A `longjmp` to a `jmp_buf` whose `setjmp`
is no longer active, because its frame has returned or it was never set, fails
with `longjmp to a jmp_buf whose setjmp is not active`. A `jmp_buf` set by
another thread is not covered by this check.

When `main` returns or the program calls `exit`, the handlers registered with
`atexit` run, and in C++ the destructors of objects with static storage
duration run after them — constructed function-local statics first, then
globals in reverse definition order, skipping namespace `std`. A static array
of class type is destroyed element by element in reverse order. An assertion in
either is therefore checked, and a global that frees memory in its destructor is
not reported as a leak ([#8339](https://github.com/esbmc/esbmc/pull/8339)).

{{< cards >}}
  {{< card link="/docs/c-cpp/supported-features" title="C++ Support" subtitle="Which C++ language and STL features ESBMC supports." >}}
  {{< card link="/docs/c-cpp/limitations" title="C++ Limitations" subtitle="What ESBMC's C++ frontend does not yet handle, and the workarounds." >}}
  {{< card link="/docs/c-cpp/esbmc-cpp-workflow-and-resources" title="C++ Workflow and Resources" subtitle="Maintainer workflow and benchmark tracking for the C++ frontend." >}}
  {{< card link="/docs/c-cpp/ctest-gen" title="CTest Test Generation" subtitle="Materialise reached witnesses as runnable CTest cases." >}}
  {{< card link="/docs/c-cpp/html-reports" title="HTML Report Generation" subtitle="Generate browsable HTML reports of verification results." >}}
  {{< card link="/docs/c-cpp/reducing-c-programs" title="Reducing C Programs" subtitle="Shrink a failing C program to a minimal reproducer." >}}
{{< /cards >}}
