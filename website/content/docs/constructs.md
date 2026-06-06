---
title: "Constructs"
date: 2026-01-20T13:34:16Z
weight: 4
draft: false
---

This page describes the verification constructs supported by ESBMC. These can be
used in harness files to aid with the verification of ESBMC.

{{< callout type="info" >}} ESBMC is also compatible with the SV-COMP
constructs. They can be used instead of these constructs. However, the ESBMC
constructs are more powerful. It is recommended to use the ESBMC constructs if
you're planning to verify your code with only ESBMC.

SV-COMP Document: https://sv-comp.sosy-lab.org/2025/rules.php {{< /callout >}}

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

## Ensure and Requires

The constructs `__ESBMC_ensure(...)`, `__ESBMC_requires(...)`,
`__ESBMC_return_value` and `__ESBMC_old(...)` are detailed in the
[Function Contracts](/docs/function-contracts) article.

## Loop Invariants

The constructs `__ESBMC_loop_invariant(...)` and `__ESBMC_loop_assigns(...)`
provide loop invariants and a frame rule (which variables a loop may change) as
an alternative to loop unwinding. They are detailed in the
[Loop Invariants](/docs/loop-invariants) article.

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

### Limitations

- Supported solvers are Z3 and CVC5 (no SMT-LIB support).
- Z3 supports only one symbol per quantifier; CVC5 supports multiple.
- Recursive quantifiers (e.g. nested `forall`) are supported.
- A constant-bounded symbol might cause incorrect simplifications (known issue).
