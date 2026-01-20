---
title: "Constructs"
date: 2026-01-20T13:34:16Z
draft: false
---

This page outlines the verification constructs that ESBMC supports. These can be
used in harness files to aid with the verification of ESBMC.

{{< callout type="info" >}}
ESBMC is compatible with the SV-COMP constructs as well. They can be used
instead of these constructs. However, the ESBMC constructs are more powerful.
It is recommended to use the ESBMC constructs if you're planning to verify your
code with only ESBMC.

SV-COMP Document: https://sv-comp.sosy-lab.org/2025/rules.php
{{< /callout >}}

## Non-Deterministic Functions

`__ESBMC_nondet_X()` where `X` is a primitive C data type. This will mark the
variable as non-deterministic, meaning it can have any value.

In this example, ESBMC will find a verification failed outcome because `x` is
marked as being able to hold any value:

```c
#include <assert.h>
int main() {
    uint x = __ESBMC_nondet_uint();
    assert(x < 10);
    return 0;
}
```

## Assert and Assume

`__ESBMC_assert(cond, reason)` can be used instead of `assert()`, this brings 
the benefit of not needing to use `#include <assert.h>`.

```c
int main() {
    uint x = __ESBMC_nondet_uint();
    __ESBMC_assert(x < 10, "X needs to be less than 10.");
    return 0;
}
```

`__ESBMC_assume(...)` can be used to narrow down the possible values of `x`.
In this case the verification will succeed because it narrows the possible
values of `x` to be less than 5.

```c
#include <assert.h>
int main() {
    uint x = __ESBMC_nondet_uint();
    __ESBMC_assume(x < 5);
    assert(x < 10);
    return 0;
}
```
## Ensure and Requires

TBD by Weiqi

## Pragma Utils

The verification paramters can be modified using `#pragma` keyword. The
following constructs are made available.

### Unroll

Unroll can be used to set the loop unwind bound for a loop. This is equivalent
to using `--unwindset id:bound` where `id` is the loop ID and `bound` is `N`.
This inlining however, allows us to specify the paramter in a more stable manner
as the `id` won't shift as the code changes. It also frees us from needing to
specify the loop bound when invoking ESBMC.

`#pragma unroll [N]` sets the next loop to be unwinded `N` times. In the
following example, the loop will be unwound 80 times max.

```c
int main() {
    uint x = __ESBMC_nondet_uint();
    __ESBMC_assume(x > 50 && x < 100);
    #pragma unroll 80
    for (int i = x - 1; x >= 0; x--) {

    }
    assert(x < 10);
    return 0;
}
```

You can also use `#pragma unroll` without `N` to make the loop unroll fully in 
the cases where `--unwind` is set. In this example, the loop will unroll fully
regardless of the global unwind bound set.

{{< callout type="warning" >}}
Be careful that the loop you use this construct to terminates, otherwise ESBMC
will never stop verifying it.
{{< /callout >}}

```c
int main() {
    uint x = __ESBMC_nondet_uint();
    __ESBMC_assume(x > 50 && x < 100);
    #pragma unroll
    for (int i = x - 1; x >= 0; x--) {

    }
    assert(x < 10);
    return 0;
}
```

`N` can also specified as a `#define` macro, however, if a value isn't found, it
will throw a parsing error.

```c
#define LOOP_BOUND 80
int main() {
    uint x = __ESBMC_nondet_uint();
    __ESBMC_assume(x > 50 && x < 100);
    #pragma unroll LOOP_BOUND
    for (int i = x - 1; x >= 0; x--) {

    }
    assert(x < 10);
    return 0;
}
```
