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

> **Experimental**: The contract annotation feature (`__ESBMC_contract`,
> `--enforce-all-contracts`, `--replace-all-contracts`) is currently available
> on the `function_contract_replace_call` branch. Checkout and build from that
> branch to use these features.

## Contract Clauses

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

### Assigns

`__ESBMC_assigns(target1, target2, ...)` specifies which memory locations a
function is allowed to modify. This clause controls the **havoc** behavior in
Replace mode:

| Assigns Clause | Havoc Behavior |
|----------------|----------------|
| `__ESBMC_assigns(target1, target2, ...)` | Havoc only the specified targets |
| `__ESBMC_assigns()` or `__ESBMC_assigns(0)` | No havoc (pure function) |
| No assigns clause | **Conservative**: havoc all static globals |

Example:

```c
int value = 0;
int other = 100;

void modify_value(void) {
    __ESBMC_assigns(value);  // Only 'value' may be modified

    value = value * 2;
}
```

When `modify_value` is replaced with its contract, only `value` is havoced.
The variable `other` retains its concrete value, allowing stronger reasoning
at the call site.

## Contract Annotation

The `__ESBMC_contract` macro marks a function for automatic contract processing.
It is a shorthand for `__attribute__((annotate("__ESBMC_contract")))`.

```c
__ESBMC_contract
void increment(void) {
    __ESBMC_requires(counter >= 0);
    __ESBMC_ensures(counter == __ESBMC_old(counter) + 1);
    __ESBMC_assigns(counter);

    counter++;
}
```

Annotated functions can then be processed in bulk with `--enforce-all-contracts`
or `--replace-all-contracts` (see [Working Logic](#working-logic) below),
instead of listing each function name individually on the command line.

### Default Contracts

When an annotated function has **no explicit** `requires` or `ensures` clauses,
ESBMC uses the following defaults:

- **Default requires**: `true` (no precondition constraint)
- **Default ensures**: `true` (no postcondition constraint)

This allows the contract machinery to process annotated functions even when they
only have an `__ESBMC_assigns` clause or no contract specification at all.

### Mixing Annotated and Non-Annotated Functions

`--enforce-all-contracts` and `--replace-all-contracts` only affect functions
marked with `__ESBMC_contract`. Non-annotated functions are left untouched:

```c
/* NOT annotated — body executes normally regardless of options */
int add(int a, int b) {
    return a + b;
}

/* Annotated — processed by --enforce-all-contracts / --replace-all-contracts */
__ESBMC_contract
void accumulate(int x) {
    __ESBMC_requires(x >= 0);
    __ESBMC_assigns(global_sum);
    __ESBMC_ensures(global_sum == __ESBMC_old(global_sum) + x);

    global_sum = global_sum + x;
}
```

## Working Logic

ESBMC processes contracts in two primary modes, depending on the verification
goal:

### Enforce (Verification)

Used to verify that a function's implementation actually satisfies its declared
contract.

| Command | Description |
|---------|-------------|
| `esbmc main.c --enforce-contract <function_name>` | Enforce a specific function's contract |
| `esbmc main.c --enforce-all-contracts` | Enforce contracts for all `__ESBMC_contract` annotated functions |

In this mode, ESBMC creates a checking wrapper that first **assumes** all
`requires` clauses, executes the function body, and finally **asserts** all
`ensures` clauses.

> **Note**: If you want to ensure the function is robust and applicable across
> various contexts without hidden dependencies, it is recommended to use the
> `--function <name>` option. In this mode, ESBMC will **havoc** parameters and
> global variables, verifying that the function satisfies its contract under all
> possible input states.

### Replace (Abstraction)

Used to substitute complex function calls with their contracts when verifying
larger programs.

| Command | Description |
|---------|-------------|
| `esbmc main.c --replace-call-with-contract <function_name>` | Replace calls to a specific function |
| `esbmc main.c --replace-all-contracts` | Replace calls to all `__ESBMC_contract` annotated functions |

When a function call is replaced:

1. ESBMC **asserts** that the pre-condition is met at the call site.
2. It **havocs** the targets specified by the `__ESBMC_assigns` clause (or all
   static globals if no assigns clause is present).
3. Finally, it **assumes** the post-condition holds.

### Combined Enforce and Replace

Both modes can be used together to verify one function while abstracting others:

```bash
esbmc main.c --enforce-contract func_a --replace-call-with-contract func_b
```

- `func_a`: Implementation verified against its contract
- `func_b`: Calls replaced with contract abstraction

> **Note**: Automatically combining enforce and replace for all annotated
> functions (i.e., using `--enforce-all-contracts` and `--replace-all-contracts`
> together) is **not currently supported**. Enforcing and replacing every
> function simultaneously provides little benefit over running ESBMC without
> contracts at all. Instead, use the per-function options to selectively enforce
> one function while replacing its callees.

### Recommended Workflow for Enforce Mode

When using `--enforce-contract` (or `--enforce-all-contracts`), it is
recommended to also specify `--function <name>` to set an explicit entry point.
This gives you control over the verification space — ESBMC will havoc
parameters and globals at the entry point and verify the contract under all
reachable states, rather than exploring the entire `main` call chain.

```bash
esbmc main.c --enforce-contract increment --function increment
```

> **Future plan**: We may provide a helper script that collects all
> `__ESBMC_contract` annotated functions from source files and generates the
> appropriate per-function ESBMC invocations, allowing modular verification
> without manual enumeration.

## Examples

### Basic Contract Verification

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

Verify with: `esbmc main.c --enforce-contract increment`

### Annotation-Based Workflow

Using `__ESBMC_contract` to mark multiple functions and verify them all at once:

```c
int counter = 0;
int value = 0;

__ESBMC_contract
void increment(void) {
    __ESBMC_requires(counter >= 0);
    __ESBMC_ensures(counter == __ESBMC_old(counter) + 1);
    __ESBMC_assigns(counter);

    counter++;
}

__ESBMC_contract
void set_value(int v) {
    __ESBMC_requires(v >= 0);
    __ESBMC_ensures(value == v);
    __ESBMC_assigns(value);

    value = v;
}

int main(void) {
    counter = 5;
    increment();
    set_value(42);
    return 0;
}
```

Verify with: `esbmc main.c --enforce-all-contracts`

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
