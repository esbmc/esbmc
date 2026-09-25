---
title: "Constructs"
date: 2026-01-20T13:34:16Z
weight: 4
draft: false
---

This page is the reference for the verification constructs — the *intrinsics* —
that ESBMC understands. They let a harness introduce non-determinism, constrain
the state space, state properties, and inspect ESBMC's memory model from inside
the program under verification.

ESBMC injects the declarations into every translation unit it parses, so **no
header is required** to call them: the declarations live in
[`clang_c_languaget::internal_additions()`](https://github.com/esbmc/esbmc/blob/master/src/clang-c-frontend/clang_c_language.cpp),
which ESBMC prepends as a synthetic `esbmc_intrinsics.h`. Because the names are
injected rather than included, a file that calls them will not compile with an
ordinary compiler — see [The `esbmc.h` Header](#the-esbmch-header) for the
supported way to keep such a file buildable.

{{< callout type="info" >}} ESBMC is also compatible with the SV-COMP
constructs. They can be used instead of these constructs. However, the ESBMC
constructs are more powerful. It is recommended to use the ESBMC constructs if
you're planning to verify your code with only ESBMC.

SV-COMP Document: https://sv-comp.sosy-lab.org/2025/rules.php {{< /callout >}}

## Quick reference

| Construct | Purpose |
| --- | --- |
| [`__ESBMC_assert(cond, msg)`](#assert-and-assume) | Report a violation when `cond` is false |
| [`__ESBMC_assume(cond)`](#assert-and-assume) | Discard executions in which `cond` is false |
| [`__ESBMC_cover(cond)`](#coverage-__esbmc_cover) | Check whether `cond` is reachable (**Python only**) |
| [`__ESBMC_unreachable()`](#unreachable-code) | Assert that control never reaches this point |
| [`nondet_X()`](#non-deterministic-functions) | A fresh non-deterministic value of type `X` |
| [`__VERIFIER_nondet_X()`](#sv-comp-compatibility) | SV-COMP spelling of the above |
| [`__VERIFIER_nondet_memory(p, n)`](#havocking-an-object) | Havoc `n` bytes at `p` |
| [`__ESBMC_init_object(p)`](#havocking-an-object) | Havoc the whole object `p` points at |
| [`__ESBMC_POINTER_OBJECT(p)`](#pointers-as-object--offset) | The object component of `p` |
| [`__ESBMC_POINTER_OFFSET(p)`](#pointers-as-object--offset) | The byte offset component of `p` |
| [`__ESBMC_same_object(p, q)`](#pointers-as-object--offset) | True when `p` and `q` address the same object |
| [`__ESBMC_get_object_size(p)`](#object-sizes) | Size in bytes of the object `p` points at |
| [`__builtin_object_size(p, t)`](#object-sizes) | GCC builtin, answered from the memory model |
| [`__ESBMC_alloc[]`, `__ESBMC_is_dynamic[]`, `__ESBMC_alloc_size[]`](#allocation-bookkeeping) | Per-object allocation bookkeeping |
| [`__ESBMC_r_ok(p, n)`](#access-validity) | True when `n` bytes at `p` may be read |
| [`__CPROVER_*`](#cbmc-compatibility-primitives) | CBMC memory primitives, mapped onto ESBMC's model |
| [`__ESBMC_memset/memcpy/memmove/memchr/memcmp`](#bulk-memory-operations) | Directly encoded bulk memory operations |
| [`__ESBMC_bitcast(tgt, src)`](#reinterpreting-bytes) | Reinterpret `*src` as the type of `*tgt` |
| [`__ESBMC_atomic_begin()` / `__ESBMC_atomic_end()`](#concurrency) | Run a region without a context switch |
| [`__ESBMC_yield()`](#concurrency) | Offer the scheduler a context switch |
| [`__ESBMC_overflow_result_*(a, b)`](#arithmetic-overflow) | Arithmetic result paired with an overflow flag |
| [`__ESBMC_rounding_mode`](#floating-point-rounding) | The current IEEE 754 rounding mode |
| [`__ESBMC_is_little_endian()`](#endianness) | True when the target is little-endian |
| [`__ESBMC_unroll(n)`](#unroll) | Bound the next loop to `n` iterations |
| [`__ESBMC_forall` / `__ESBMC_exists`](#quantifiers) | Quantified predicates |
| [`__ESBMC_requires` / `__ESBMC_ensures` / `__ESBMC_assigns` / `__ESBMC_old` / `__ESBMC_return_value` / `__ESBMC_is_fresh`](#function-contracts-and-loop-invariants) | Function contracts |
| [`__ESBMC_loop_invariant` / `__ESBMC_loop_assigns`](#function-contracts-and-loop-invariants) | Loop contracts |
| [`__ESBMC_and` / `__ESBMC_or` / `__ESBMC_implies`](#non-short-circuiting-connectives) | Non-short-circuiting connectives |
| [`__esbmc_cheri_*`](#cheri-c) | CHERI capability accessors (CHERI builds only) |

## The `esbmc.h` Header

ESBMC installs an `esbmc.h` header alongside the binary. Including it gives the
intrinsics unprefixed names, which is the supported surface for hand-written
harnesses and stubs:

| Macro                       | Expands to                  |
| --------------------------- | --------------------------- |
| `ESBMC_assume(cond)`        | `__ESBMC_assume`            |
| `ESBMC_assert(cond, msg)`   | `__ESBMC_assert`            |
| `ESBMC_alloca(size)`        | `__ESBMC_alloca`            |
| `ESBMC_same_object(p, q)`   | `__ESBMC_same_object`       |
| `ESBMC_unreachable()`       | `__ESBMC_unreachable`       |
| `ESBMC_unroll(n)`           | `__ESBMC_unroll`            |
| `ESBMC_atomic_begin()` / `ESBMC_atomic_end()` | `__ESBMC_atomic_begin` / `__ESBMC_atomic_end` |
| `ESBMC_yield()`             | `__ESBMC_yield`             |

Every ESBMC run defines `__ESBMC_execution`, and the header refuses to compile
without it — so a harness that includes `esbmc.h` is a hard error under any
ordinary compiler instead of silently building with the intrinsics undefined.
The same macro lets a source tree tell an ESBMC run from an ordinary compile:

```c
#ifdef __ESBMC_execution
#  include <esbmc.h>
#endif

int f(unsigned len) {
#ifdef __ESBMC_execution
    ESBMC_assume(len < 16);   /* only under verification */
#endif
    return len;
}
```

The `#include` has to sit inside the guard as well: the header `#error`s when
`__ESBMC_execution` is undefined, so an unguarded include would break exactly
the ordinary compile the guard exists to protect. Code that is compiled only by
ESBMC can use the `__ESBMC_`-prefixed intrinsics directly and skip the header.

`__ESBMC_alloca` is the one entry in that table that is not an injected
declaration: ESBMC passes `-D__ESBMC_alloca=__builtin_alloca` to clang, so it
allocates with the lifetime of the enclosing function.

```c
#include <esbmc.h>

int main() {
    unsigned int x = nondet_uint();
    ESBMC_assume(x < 5);
    ESBMC_assert(x < 10, "X needs to be less than 10.");
    return 0;
}
```

## Non-Deterministic Functions

`nondet_X()` where `X` is a primitive C data type. This will mark the variable
as non-deterministic, meaning it can have any value.

In this example, ESBMC will find a verification failed outcome because `x` is
marked as being able to hold any value:

```c
#include <assert.h>
int main() {
    unsigned int x = nondet_uint();
    assert(x < 10);
    return 0;
}
```

`X` ranges over `bool`, `char`, `schar`, `uchar`, `short`, `ushort`, `int`,
`uint`, `long`, `ulong`, `float` and `double`. More generally, **any function
whose body is unavailable returns a fresh non-deterministic value of its return
type**, so an external function that is declared but never defined behaves like
a `nondet_` call. See [Modeling with
Non-determinism](/docs/theory/non-determinism) for the underlying model.

### Havocking an object

Two intrinsics discard what ESBMC knows about memory that already exists, which
is how a harness models "some other component wrote here".

```c
void __ESBMC_init_object(void *p);                        /* whole object */
void __VERIFIER_nondet_memory(void *p, __SIZE_TYPE__ n);  /* n bytes */
```

`__ESBMC_init_object(p)` assigns a fresh non-deterministic value of the object's
own type to the *entire* object `p` points at, so any value previously stored
there is forgotten:

```c
#include <assert.h>
int main() {
    int v[4];
    v[3] = 5;
    __ESBMC_init_object(v);
    assert(v[3] == 5);   /* fails: the object was havoc'd */
    return 0;
}
```

It needs a statically known object size: a VLA or an infinite-length array is
rejected with a diagnostic. `__VERIFIER_nondet_memory(p, n)` is the byte-granular
alternative — it writes `n` non-deterministic bytes and so can havoc part of an
object, or an object whose type is not known at the call. Note that a reachable
call to it disables k-induction's inductive step.

## Assert and Assume

`__ESBMC_assert(cond, reason)` can be used instead of `assert()`, this brings
the benefit of not needing to use `#include <assert.h>`.

```c
int main() {
    unsigned int x = nondet_uint();
    __ESBMC_assert(x < 10, "X needs to be less than 10.");
    return 0;
}
```

`__ESBMC_assume(int)` can be used to narrow down the possible values of `x`. In
this case, the verification will succeed because it narrows the possible values
of `x` to be less than 5.

```c
#include <assert.h>
int main() {
    unsigned int x = nondet_uint();
    __ESBMC_assume(x < 5);
    assert(x < 10);
    return 0;
}
```

{{< callout type="warning" >}} An assumption that is false on every path makes
verification **vacuous**: everything after it is unreachable, so every property
"passes" without being tested. Pair a strong assumption with a reachability
check — an `__ESBMC_unreachable()` that is *expected* to fail, or
`__ESBMC_cover` on a Python harness. {{< /callout >}}

### Unreachable code

`__ESBMC_unreachable()` claims that control never reaches the call. It is the
primitive behind unreach-call style checking, and behind proving that a branch
is dead.

```c
_Noreturn void __ESBMC_unreachable(void);
```

The claim is **only raised under `--enable-unreachability-intrinsic`**; without
that flag the call is a no-op and verification reports success even where the
site is plainly reachable:

```c
int main() {
    int x = nondet_int();
    if (x > 0)
        __ESBMC_unreachable();   /* reachable */
    return 0;
}
```

```sh
esbmc file.c                                     # VERIFICATION SUCCESSFUL (claim suppressed)
esbmc file.c --enable-unreachability-intrinsic   # VERIFICATION FAILED
```

A violation reports `reachability: unreachable code reached` and names the
source location. The same flag makes `reach_error()` and `__VERIFIER_error()`
error sentinels, so a call to either is reported even when the program provides
its own body for them.

### Coverage: `__ESBMC_cover`

`__ESBMC_cover(cond)` asks the opposite question to `assert`: it checks whether
`cond` is *satisfiable* at that point, so a counterexample means the condition
**is** reachable. Use it with `--multi-property`.

{{< callout type="warning" >}} `__ESBMC_cover` is declared for C but implemented
only in the [Python frontend](/docs/python/usage). Calling it from C or C++ is a
fatal error: `Function call to non-intrinsic prefixed with __ESBMC`. In C, use
an intentionally failing `__ESBMC_assert(!cond, ...)`, or
`__ESBMC_unreachable()` under `--enable-unreachability-intrinsic`. {{< /callout >}}

```python
if x > 100:
    __ESBMC_cover(x > 100)   # is this branch reachable?
```

## SV-COMP compatibility

ESBMC accepts the SV-COMP vocabulary directly, so a benchmark written for the
competition needs no adaptation:

| SV-COMP | ESBMC equivalent |
| --- | --- |
| `__VERIFIER_nondet_X()` | `nondet_X()` |
| `__VERIFIER_assume(int)` | `__ESBMC_assume` |
| `__VERIFIER_atomic_begin()` / `__VERIFIER_atomic_end()` | `__ESBMC_atomic_begin` / `__ESBMC_atomic_end` |
| `__VERIFIER_error()` | Asserts false at the call (see below) |
| `__VERIFIER_nondet_memory(p, n)` | (no ESBMC-prefixed spelling) |

`__VERIFIER_error()` and `reach_error()` assert false at the call site on their
own — no flag required. What `--enable-unreachability-intrinsic` adds is that a
body the *program itself* supplies for either name is skipped, so the violation
is reported at the call rather than inside that body. Without the flag, a
program that defines its own `__VERIFIER_error` gets its definition called and
no assertion:

```c
#include <stdio.h>
void __VERIFIER_error(void) { printf("boom\n"); }   /* own body */
int main() { int x = nondet_int(); if (x == 42) __VERIFIER_error(); return 0; }
```

```sh
esbmc file.c                                     # VERIFICATION SUCCESSFUL (body runs)
esbmc file.c --enable-unreachability-intrinsic   # VERIFICATION FAILED (sentinel)
```

## The memory model

The intrinsics below expose the representation described in [Memory Model and
Pointer Safety](/docs/theory/memory-model). They are what ESBMC's own
operational models are written against, and they are equally available to a
harness that needs to state a memory-shaped precondition.

### Pointers as object + offset

```c
__UINTPTR_TYPE__ __ESBMC_POINTER_OBJECT(const void *p);
__PTRDIFF_TYPE__ __ESBMC_POINTER_OFFSET(const void *p);
_Bool            __ESBMC_same_object(const void *p, const void *q);
```

A pointer is a pair of *which object* and *what byte offset into it*. Pointer
arithmetic moves the offset and keeps the object fixed, so `p` and `p + i`
always share an object:

```c
#include <assert.h>
int main() {
    char a[10];
    char *p = a + 3;
    assert(__ESBMC_same_object(a, p));
    assert(__ESBMC_POINTER_OFFSET(p) == 3);
    assert(__ESBMC_POINTER_OBJECT(a) == __ESBMC_POINTER_OBJECT(p));
    return 0;
}
```

### Object sizes

```c
__SIZE_TYPE__ __ESBMC_get_object_size(const void *p);
```

`__ESBMC_get_object_size(p)` yields the number of *elements* in the array object
`p` points into, regardless of `p`'s offset. GCC's `__builtin_object_size(p, type)`
is rewritten to `__ESBMC_builtin_object_size` and answered from the same model
rather than from a compile-time approximation, so it stays exact for heap
objects. It counts **bytes**, and it measures the object addressed rather than
`sizeof(*p)`: a scalar reached through a `void *` or a `char *` reports the
scalar's own size, not `0` or `1`. The two therefore agree only where the
element is one byte wide:

```c
#include <assert.h>
#include <stdlib.h>
int main() {
    char a[10];
    int n[4];
    char *d = malloc(8);
    assert(__ESBMC_get_object_size(a) == 10);
    assert(__builtin_object_size(a, 0) == 10);
    assert(__ESBMC_get_object_size(n) == 4);    /* elements */
    assert(__builtin_object_size(n, 0) == 16);  /* bytes */
    assert(__ESBMC_get_object_size(d) == 8);

    void *v = &n[0];
    assert(__builtin_object_size(v, 0) == 16);  /* the object, not sizeof(void) */
    return 0;
}
```

### `offsetof`

In C, ESBMC defines `__builtin_offsetof(type, member)` in terms of the pointer
decomposition above:

```c
#define __builtin_offsetof(type, member) \
    ((size_t)__ESBMC_POINTER_OFFSET(&((type*)0)->member))
```

{{< callout type="warning" >}} That expansion is *not* an integer constant
expression, although C23 7.21p3 requires `offsetof` to be one. So under the C
frontend `offsetof` may not be spelt where a constant expression is required —
an array bound, a `case` label, an enumerator. Used as an array bound at file
scope it is reported as `variable length array declaration not allowed at file
scope`. Every runtime use is unaffected, as is C++, where clang's own
`OffsetOfExpr` is lowered directly instead. {{< /callout >}}

### Allocation bookkeeping

Three infinite-length arrays record the state of every object, indexed by the
object number that `__ESBMC_POINTER_OBJECT` returns:

```c
extern _Bool         __ESBMC_alloc[];       /* is this dynamic object still allocated? */
extern _Bool         __ESBMC_is_dynamic[];  /* was it heap-allocated? */
extern __SIZE_TYPE__ __ESBMC_alloc_size[];  /* the size it was allocated with */
```

Each is declared with ESBMC's `__ESBMC_inf_size` annotation, so it has an entry
for every object the program can create rather than a fixed bound.

```c
#include <assert.h>
#include <stdlib.h>
int main() {
    char a[10];
    char *d = malloc(8);
    assert(__ESBMC_is_dynamic[__ESBMC_POINTER_OBJECT(d)]);
    assert(!__ESBMC_is_dynamic[__ESBMC_POINTER_OBJECT(a)]);
    assert(__ESBMC_alloc[__ESBMC_POINTER_OBJECT(d)]);
    assert(__ESBMC_alloc_size[__ESBMC_POINTER_OBJECT(d)] == 8);
    return 0;
}
```

{{< callout type="info" >}} `__ESBMC_alloc` tracks *dynamic* allocation only. It
stays false for a live static or automatic object, so liveness may only be read
off it for objects that `__ESBMC_is_dynamic` flags. {{< /callout >}}

### Access validity

```c
_Bool __ESBMC_r_ok(void *p, unsigned long n);
```

True when reading `n` bytes from `p` stays inside the object `p` points into. A
zero-length access is always in bounds:

```c
#include <assert.h>
int main() {
    char arr[10];
    assert(__ESBMC_r_ok(arr, 10));
    assert(__ESBMC_r_ok(arr, 0));
    return 0;
}
```

## CBMC compatibility primitives

ESBMC implements CBMC's memory primitives on top of the model above, so a
program written against CBMC verifies against the real memory model instead of
against havoc'd values ([#2457](https://github.com/esbmc/esbmc/issues/2457)):

| Primitive | Meaning |
| --- | --- |
| `__CPROVER_POINTER_OBJECT(p)` | Object component of `p` |
| `__CPROVER_POINTER_OFFSET(p)` | Byte offset component of `p` |
| `__CPROVER_same_object(p, q)` | `p` and `q` address the same object |
| `__CPROVER_OBJECT_SIZE(p)` | Size in bytes of the object addressed, whatever `p`'s own type (`0` for `NULL`) |
| `__CPROVER_DYNAMIC_OBJECT(p)` | `p` points into heap-allocated memory |
| `__CPROVER_LIVE_OBJECT(p)` | The object is still allocated |
| `__CPROVER_WRITEABLE_OBJECT(p)` | Coincides with `__CPROVER_LIVE_OBJECT` |
| `__CPROVER_r_ok(p, n)` | `n` bytes at `p` may be read |
| `__CPROVER_w_ok(p, n)` | Coincides with `__CPROVER_r_ok` |
| `__CPROVER_rw_ok(p, n)` | Coincides with `__CPROVER_r_ok` |

ESBMC's memory is uniformly readable and writeable, which is why `w_ok` and
`rw_ok` coincide with `r_ok`.

```c
#include <assert.h>
#include <stdlib.h>
int main() {
    char a[4];
    char *d = malloc(8);
    assert(__CPROVER_OBJECT_SIZE(a) == 4);
    assert(!__CPROVER_DYNAMIC_OBJECT(a));
    assert(__CPROVER_DYNAMIC_OBJECT(d));
    assert(__CPROVER_r_ok(a, 4));
    assert(!__CPROVER_r_ok(a, 8));
    return 0;
}
```

Run this one with `--force-malloc-success`: by default ESBMC also explores the
path on which `malloc` returns `NULL`, and `__CPROVER_DYNAMIC_OBJECT(NULL)` is
false by definition.

## Bulk memory operations

```c
void *__ESBMC_memset(void *s, int c, __SIZE_TYPE__ n);
void *__ESBMC_memcpy(void *dst, const void *src, __SIZE_TYPE__ n);
void *__ESBMC_memmove(void *dst, const void *src, __SIZE_TYPE__ n);
void *__ESBMC_memchr(const void *buf, int ch, __SIZE_TYPE__ n);
int   __ESBMC_memcmp(const void *s1, const void *s2, __SIZE_TYPE__ n);
```

These are the entry points that `<string.h>`'s `memset`, `memcpy`, `memmove`,
`memchr` and `memcmp` call, so ordinary code reaches them without naming them.
ESBMC encodes each one directly rather than unwinding a byte loop — for example
`memchr` folds the candidate bytes into a single nested-`ite` pointer expression
instead of branching per byte. That optimisation needs a **constant** length;
when `n` is non-deterministic ESBMC falls back to the loop-based operational
model, which is correct but costs an unwind bound.

Call them by name when you want the encoded form regardless of what the program
includes:

```c
#include <assert.h>
int main() {
    char b[4], c[4];
    __ESBMC_memset(b, 0xAB, 4);
    assert(b[0] == (char)0xAB && b[3] == (char)0xAB);
    __ESBMC_memcpy(c, b, 4);
    assert(__ESBMC_memcmp(c, b, 4) == 0);
    return 0;
}
```

### Reinterpreting bytes

```c
void __ESBMC_bitcast(void *tgt, void *src);
```

Equivalent to `memcpy(tgt, src, n)` where `n` is the size of the type the two
pointers point at — the type-punning primitive ESBMC's own float models use:

```c
#include <assert.h>
#include <stdint.h>
int main() {
    float f = 1.0f;
    uint32_t u;
    __ESBMC_bitcast(&u, &f);
    assert(u == 0x3f800000);
    return 0;
}
```

{{< callout type="warning" >}} Two preconditions are checked by assertions
inside ESBMC, so violating either **aborts the run** rather than producing a
diagnostic or a counterexample:

- **The two pointee types must have the same width.** `__ESBMC_bitcast(&u, &d)`
  for a 4-byte `u` and an 8-byte `d` trips
  `type->get_width() == from_->type->get_width()` in `bitcast2t`.
- **Each argument must resolve to exactly one object.** A pointer that may
  address either of two objects trips `internal_deref_items.size() == 1` in
  `run_intrinsic`.

Prefer `memcpy` when either condition is not statically guaranteed.
{{< /callout >}}

## Concurrency

```c
void __ESBMC_atomic_begin(void);
void __ESBMC_atomic_end(void);
void __ESBMC_yield(void);
```

Statements between `__ESBMC_atomic_begin()` and `__ESBMC_atomic_end()` execute
without an intervening context switch, so the region is indivisible with respect
to other threads. `__ESBMC_yield()` does the opposite: it offers the scheduler a
switch at that point.

```c
#include <pthread.h>
#include <assert.h>
int x = 0;
void *t(void *a) {
    __ESBMC_atomic_begin();
    x++; x++;                /* other threads never observe the odd state */
    __ESBMC_atomic_end();
    return 0;
}
int main() {
    pthread_t p;
    pthread_create(&p, 0, t, 0);
    __ESBMC_atomic_begin();
    assert(x % 2 == 0);
    __ESBMC_atomic_end();
    pthread_join(p, 0);
    return 0;
}
```

Removing the atomic section makes the assertion fail, because the interleaving
that observes `x == 1` becomes reachable. See
[Concurrency](/docs/theory/concurrency) for the scheduling model.

## Arithmetic overflow

`__ESBMC_overflow_result_*` computes an arithmetic result together with a flag
saying whether the operation overflowed, in the style of GCC's
`__builtin_*_overflow`. The result type is per-operand-type, so instantiate it
first with `DEFINE_ESBMC_OVERFLOW_TYPE`:

```c
#include <assert.h>
DEFINE_ESBMC_OVERFLOW_TYPE(int)

int main() {
    __ESBMC_overflow_result r = __ESBMC_overflow_result_plus(2147483647, 1);
    assert(r.overflow);

    __ESBMC_overflow_result s = __ESBMC_overflow_result_plus(1, 2);
    assert(!s.overflow && s.result == 3);
    return 0;
}
```

The macro declares `__ESBMC_overflow_result` as a packed
`struct { _Bool overflow; type result; }` together with
`__ESBMC_overflow_result_plus`, `_minus`, `_mult`, `_shl` (all binary) and
`_unary_minus`.

{{< callout type="warning" >}} The typedef name is fixed, so **at most one
`DEFINE_ESBMC_OVERFLOW_TYPE` per translation unit**. A second one is a
`typedef redefinition with different types` error — including a second
instantiation for the *same* type, since each expansion defines a fresh
anonymous struct. To check more than one operand type, split the checks across
translation units. {{< /callout >}}

## Floating point and endianness

### Floating-point rounding

```c
extern int __ESBMC_rounding_mode;
```

Floating-point operations read their rounding mode from this global. It is an
ordinary variable, so a harness may set it directly, and `<fenv.h>`'s
`fesetround`/`fegetround` are modelled on top of it:

| Value | Rounding mode | `<fenv.h>` |
| --- | --- | --- |
| `0` | To nearest, ties to even (default) | `FE_TONEAREST` |
| `2` | Toward `+∞` | `FE_UPWARD` |
| `3` | Toward `−∞` | `FE_DOWNWARD` |
| `4` | Toward zero | `FE_TOWARDZERO` |

```c
#include <assert.h>
#include <fenv.h>
int main() {
    assert(__ESBMC_rounding_mode == 0);
    fesetround(FE_UPWARD);
    assert(__ESBMC_rounding_mode == 2);
    return 0;
}
```

The `--round-to-nearest`, `--round-to-plus-inf`, `--round-to-minus-inf` and
`--round-to-zero` flags set the same global for the whole run.

### Endianness

```c
_Bool __ESBMC_is_little_endian(void);
```

Reports the byte order of the configured target, so an endianness-dependent
model can branch on it. It follows `--big-endian` / `--little-endian`.

## Pragma Utils

The verification paramters can be modified using `#pragma` keyword. The
following constructs are made available.

### Unroll

Unroll can be used to set the loop unwind bound for a loop. This is equivalent
to using `--unwindset id:bound` where `id` is the loop ID and `bound` is `N`.
This inlining, however, allows us to specify the parameter in a more stable
manner as the `id` won't shift as the code changes. It also frees us from
needing to specify the loop bound when invoking ESBMC.

`#pragma unroll [N]` sets the next loop to be unwound `N` times. In the
following example, the loop will be unwound 80 times max.

```c
int main() {
    unsigned int x = nondet_uint();
    __ESBMC_assume(x > 50 && x < 100);
    unsigned int y = 0;
    #pragma unroll 80
    for (int i = x - 1; x >= 0; x--) {
        y += x;
    }
    assert(y > 100);
    return 0;
}
```

You can also use `#pragma unroll` without `N` to make the loop unroll fully in
the cases where `--unwind` is set. In this example, the loop will unroll fully
regardless of the global unwind bound set.

{{< callout type="warning" >}} Be careful that the loop you use this construct
to terminate, otherwise ESBMC will never stop verifying it. {{< /callout >}}

```c
int main() {
    unsigned int x = nondet_uint();
    __ESBMC_assume(x > 50 && x < 100);
    #pragma unroll
    for (int i = x - 1; x >= 0; x--) {
        y += x;
    }
    assert(y > 100);
    return 0;
}
```

`N` can also be specified as a `#define` macro; however, if a value isn't found,
it will throw a parsing error.

```c
#define LOOP_BOUND 80
int main() {
    unsigned int x = nondet_uint();
    __ESBMC_assume(x > 50 && x < 100);
    #pragma unroll LOOP_BOUND
    for (int i = x - 1; x >= 0; x--) {
        y += x;
    }
    assert(y > 100);
    return 0;
}
```

Alternatively, the same behavior can be obtained through the
`__ESBMC_unroll(LOOP_BOUND)` intrinsic.

```c
#define LOOP_BOUND 80
int main() {
    unsigned int x = nondet_uint();
    __ESBMC_assume(x > 50 && x < 100);
    __ESBMC_unroll(LOOP_BOUND);
    for (int i = x - 1; x >= 0; x--) {
        y += x;
    }
    assert(y > 100);
    return 0;
}
```

The intrinsic must be placed immediately before the loop it applies to. Only the
loop's own setup (the declarations and initialisers of a `for` loop, or the
declaration in a condition such as `while (int v = f())`) may appear between the
intrinsic and the loop header. It binds to the nearest following loop, so for
nested loops it annotates the inner loop:

```c
while (1) {
    __ESBMC_unroll(10);
    for (int i = 0, j = 10; i < j; i++, j--) // annotated with 10
        ;
}
```

If an `__ESBMC_unroll` call is not directly followed by a loop (for example, an
unrelated statement is placed in between), ESBMC reports a warning and ignores
the annotation.

## Quantifiers

ESBMC supports universal (`forall`) and existential (`exists`) quantifiers in
SMT-based verification. Two expressions are available:

- `bool forall(symbol, predicate)` — holds if the predicate holds for all values
  of `symbol`.
- `bool exists(symbol, predicate)` — holds if the predicate holds for at least
  one value of `symbol`.

They are declared as:

```c
extern void __ESBMC_assume(_Bool);
extern _Bool __ESBMC_forall(void *, _Bool);
extern _Bool __ESBMC_exists(void *, _Bool);
```

### Example

```c
int main() {
  unsigned n;
  int arr[n];
  unsigned i;

  __ESBMC_assume(__ESBMC_forall(&i, !(i < n) || arr[i] == 2));
  __ESBMC_assert(!__ESBMC_exists(&i, (i < n) && arr[i] == 42), "forall init");

  arr[n/2] = 42;
  __ESBMC_assert(!__ESBMC_exists(&i, (i < n) && arr[i] == 42), "this should fail");
}
```

```c
int zero_array[10];
int main() {
  int sym;
  __ESBMC_assert(
    __ESBMC_forall(&sym, !(sym >= 0 && sym < 10) || zero_array[sym] == 0),
    "array is zero initialized");

  const unsigned N = 10;
  char c[N];
  for (unsigned i = 0; i < N; ++i) c[i] = i;

  unsigned j;
  __ESBMC_assert(__ESBMC_forall(&j, j > 9 || c[j] == j),
    "array is initialized correctly");
}
```

Run with a supported solver:

```sh
esbmc file.c --z3
```

### Calls, branches and loops inside a quantifier body

A quantifier body may call a function. ESBMC inlines the callee *under* the
binder, so the bound variable stays free:

```c
_Bool eq(int a, int b) { return a == b; }

int main() {
  int var;
  __ESBMC_assert(__ESBMC_forall(&var, eq(var, 6)), "not valid for every var");
}
```

Beyond a single `return`, a callee built from local declarations, assignments,
`if`/`else`, and loops with a **statically constant** trip count is summarized
into one side-effect-free expression — so a counting or accumulating helper can
appear under `__ESBMC_forall`/`__ESBMC_exists` directly.

Summarization is bounded by the size of the resulting expression rather than by
the iteration count, because a branch merge inside an unrolled loop can double
the summary on every iteration. The cap is
`--max-quantifier-summary-nodes NR` (default 20000); raise it if a legitimate
body is rejected for size.

Shapes that cannot be summarized soundly — data-dependent trip counts, pointer
writes, `break`, `switch`, recursion — are **rejected with a diagnostic naming
the cause**, not silently hoisted out of the binder (which would freeze the
bound variable and make the quantifier vacuous).

### Limitations

- Supported solvers are Z3 and CVC5 (no SMT-LIB support).
- Z3 supports only one symbol per quantifier; CVC5 supports multiple.
- Recursive quantifiers (e.g. nested `forall`) are supported.
- A constant-bounded symbol might cause incorrect simplifications (known issue).

## Function contracts and loop invariants

These constructs have pages of their own:

| Construct | Documented in |
| --- | --- |
| `__ESBMC_requires`, `__ESBMC_ensures`, `__ESBMC_assigns`, `__ESBMC_old`, `__ESBMC_return_value`, `__ESBMC_is_fresh`, `__ESBMC_contract` | [Function Contracts](/docs/function-contracts) |
| `__ESBMC_loop_invariant`, `__ESBMC_loop_assigns` | [Loop Invariants](/docs/loop-invariants) |

Contract annotations are always declared, so an annotated file compiles under
any ESBMC mode; in plain BMC they are dropped, and enforcement only starts with
`--enforce-contract` or `--replace-call-with-contract`.

### Non-short-circuiting connectives

```c
#define __ESBMC_and(a, b)     ((a) & (b))
#define __ESBMC_or(a, b)      ((a) | (b))
#define __ESBMC_implies(a, b) ((!(a)) | (b))
```

`&&` and `||` short-circuit, which introduces control flow into what a contract
clause needs to be a single side-effect-free expression. These macros are the
bitwise, unconditionally-evaluating forms, plus the implication that C lacks:

```c
__ESBMC_requires(__ESBMC_implies(n > 0, p != 0));
```

## CHERI-C

When ESBMC is built with CHERI support and run with `--cheri hybrid` or
`--cheri purecap`, a further set of capability accessors is declared:

```c
__SIZE_TYPE__     __esbmc_cheri_length_get(void *__capability);
__SIZE_TYPE__     __esbmc_cheri_base_get(void *__capability);
void *__capability __esbmc_cheri_bounds_set(void *__capability, __SIZE_TYPE__);
```

With the 128-bit compressed format (`__ESBMC_CHERI__ == 128`) the accessors for
the remaining capability fields are also available: `__esbmc_cheri_top_get`,
`__esbmc_cheri_perms_get`, `__esbmc_cheri_flags_get`, `__esbmc_cheri_type_get`
and `__esbmc_cheri_sealed_get`.

The same block declares `__esbmc_clzll` (count leading zeros, used by the
capability-decoding model) and `__ESBMC_cheri_info`, an infinite-size array of
`{base, top}` records tracking the bounds ESBMC has seen set.

`__esbmc_cheri_bounds_set` follows the CSetBounds semantics: it asserts that the
capability is tagged and that its base does not exceed the cursor, narrows the
bounds, and preserves the object and offset of the input capability.

## Internal intrinsics

The names below are declared in the same header but are **not** part of the
supported surface for user code — they exist so ESBMC's own operational models
and instrumentation passes can be type-checked and linked. They are listed here
so that a name encountered in a counterexample or a GOTO dump can be identified.

| Name | Role |
| --- | --- |
| `__ESBMC_pthread_start_main_hook`, `__ESBMC_pthread_end_main_hook` | Main-thread begin/end hooks pulled in from the C library |
| `__ESBMC_atexit_handler` | Runs functions registered with `atexit` |
| `__ESBMC_memory_leak_checks` | Emits the memory-leak claims at program exit |
| `__ESBMC_witness_assume` | Injected by the witness validator to replay an assumption waypoint |
| `__ESBMC_assigns_impl`, `__ESBMC_loop_assigns_impl` | Variadic backends behind the `__ESBMC_assigns` / `__ESBMC_loop_assigns` macros, which dispatch on argument count and accept **at most five** targets |
| `__ESBMC_old_raw` | Backend behind the `__ESBMC_old` macro |
