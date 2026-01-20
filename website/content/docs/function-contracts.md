---
title: "Function Contracts"
date: 2026-01-20T13:34:16Z
draft: false
weight: 4
---

Function contracts allow developers to define logical specifications between a
function and its callers. ESBMC uses these contracts to verify the correctness
of a function's implementation or to simplify function calls in complex
verification tasks by using the contract as an abstraction.

### Requires and Ensures

- `__ESBMC_requires(cond)`: Defines the **pre-condition** that must be met
  before the function is executed.
- `__ESBMC_ensures(cond)`: Defines the **post-condition** that must be
  guaranteed to hold after the function completes.

Inside the post-condition (`ensures`), you can use the following special
constructs:

- `__ESBMC_return_value`: Represents the value returned by the function.
- `__ESBMC_old(expr)`: Represents the initial value of an expression at the
  moment the function was entered (a snapshot).

### Working Logic

ESBMC processes contracts in two primary modes, depending on the verification
goal:

#### Enforce (Verification)

Used to verify that a function's implementation actually satisfies its declared
contract. You can trigger this mode using the following command:

`esbmc main.c --enforce-contract <function_name>`

In this mode, ESBMC creates a checking wrapper that first **assumes** all
`requires` clauses, executes the function body, and finally **asserts** all
`ensures` clauses.

> **Note**: If you want to ensure the function is robust and applicable across
> various contexts without hidden dependencies, it is recommended to use the
> `--function <name>` option. In this mode, ESBMC will **havoc** parameters and
> global variables, verifying that the function satisfies its contract under all
> possible input states.

#### Replace (Abstraction)

Used to substitute complex function calls with their contracts when verifying
larger programs. You can trigger this mode using the following command:

`esbmc main.c --replace-calls-with-contracts <function_name>`

When a function call is replaced:

1. ESBMC **asserts** that the pre-condition is met at the call site.
2. It then **havocs** all global variables. This is a conservative approach to
   ensure that any potential side effects of the original function are accounted
   for.
3. Finally, it **assumes** the post-condition holds.

This **havoc** mechanism ensures that the function's influence on the system
state is correctly propagated from the removed function body into the `main`
function or the rest of the call chain, even if the specific modification
footprint is not precisely analyzed.

### Example Usage

In this example, we use `__ESBMC_old` to ensure a global variable is correctly
incremented and `__ESBMC_return_value` to verify the return logic:

```c
#include <assert.h>

int count = 0;

int increment(int n) {
    // Pre-condition
    __ESBMC_requires(n < 100);

    // Post-conditions
    __ESBMC_ensures(__ESBMC_return_value == n + 1);
    __ESBMC_ensures(count == __ESBMC_old(count) + 1);

    count++;
    return n + 1;
}

int main() {
    int val = __ESBMC_nondet_int();
    __ESBMC_assume(val < 10);
    increment(val);
    return 0;
}

```

## Memory Freshness

`__ESBMC_is_fresh(ptr, size)` is a special construct used to verify memory
allocation. It is typically used in `ensures` clauses to check if a pointer
`ptr` points to a valid block of dynamic memory of `size` bytes that was newly
allocated within the function.

```c
void allocate_buffer(char **ptr) {
    // Ensures that after the function, *ptr points to a fresh 10-byte allocation
    __ESBMC_ensures(__ESBMC_is_fresh(*ptr, 10));
    *ptr = malloc(10);
}

```

