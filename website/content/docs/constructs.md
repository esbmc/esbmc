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


