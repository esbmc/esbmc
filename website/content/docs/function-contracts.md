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

{{< callout type="warning" >}}
**Experimental**: The contract annotation feature (`__ESBMC_contract`,
`--enforce-all-contracts`, `--replace-all-contracts`) is currently available
on the `function_contract_replace_call` branch. Checkout and build from that
branch to use these features.
{{< /callout >}}

## Contract Clauses

ESBMC provides the following constructs for specifying function contracts:

| Construct | Purpose |
|-----------|---------|
| `__ESBMC_requires(cond)` | Pre-condition |
| `__ESBMC_ensures(cond)` | Post-condition |
| `__ESBMC_assigns(targets...)` | Modification frame |
| `__ESBMC_is_fresh(ptr, size)` | Memory freshness |
| `__ESBMC_old(expr)` | Snapshot of entry-time value (inside `ensures`) |
| `__ESBMC_return_value` | Function return value (inside `ensures`) |

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
Replace mode.

**Havocing** means assigning a nondeterministic (arbitrary) value to a variable.
When ESBMC replaces a function call with its contract, it havocs the specified
targets to model the fact that those memory locations may have been modified by
the function in any way consistent with the post-condition.

Summary of havoc behavior:

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

### Memory Freshness

`__ESBMC_is_fresh(ptr, size)` is a special construct used in contract clauses
to reason about dynamically allocated memory. It asserts that `ptr` points to
a valid block of at least `size` bytes that was freshly allocated (i.e., not
aliasing any pre-existing memory).

**Usage in contracts:**

- In `__ESBMC_requires`: assume the caller provides a freshly allocated pointer
  of the given size as input to the function.
- In `__ESBMC_ensures`: assert that after the function returns, the pointer
  refers to a newly allocated memory block of the given size.

```c
void allocate_buffer(char **ptr) {
    __ESBMC_ensures(__ESBMC_is_fresh(*ptr, 10));
    *ptr = malloc(10);
}

void fill_buffer(char *buf, int len) {
    __ESBMC_requires(__ESBMC_is_fresh(buf, len));
    __ESBMC_requires(len > 0);
    __ESBMC_ensures(buf[0] == 'A');
    buf[0] = 'A';
}
```

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

### Default Contracts

When an annotated function has **no explicit** `requires` or `ensures` clauses,
ESBMC uses the following defaults:

- **Default requires**: `true` (no precondition constraint)
- **Default ensures**: `true` (no postcondition constraint)

This allows the contract machinery to process annotated functions even when they
only have an `__ESBMC_assigns` clause or no contract specification at all.

**What happens when an annotated function has no contract clauses at all?**

- **Enforce mode**: ESBMC assumes `true` as the precondition, executes the
  function body, and asserts `true` as the postcondition. The postcondition
  check is trivially satisfied, but any `assert` statements inside the function
  body are still verified normally.
- **Replace mode**: ESBMC asserts `true` (always passes), havocs all static
  globals (since no `__ESBMC_assigns` is present), and assumes `true` (no
  constraint on the havoced values). This means every global variable becomes
  nondeterministic with no postcondition to constrain them. If the havoced
  globals are not used in subsequent verification (e.g., the function's side
  effects do not affect `main`), the verification may still pass. Otherwise,
  the unconstrained nondeterminism is likely to produce **false positives**.
  Adding an `__ESBMC_assigns` clause — even without `requires`/`ensures` — can
  significantly reduce false positives by limiting havoc to only the relevant
  targets.

In general, annotated functions should include meaningful contract clauses to
get useful verification results.

{{< callout type="info" >}}
Compositional verification introduces additional proof obligations — each
function must be individually verified against its contract. In replace mode,
the callee's body is not unrolled at the call site, so the performance benefit
grows with the number of call sites: the more often a function is called, the
more redundant analysis is avoided.
{{< /callout >}}

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

{{< callout type="info" >}}
It is recommended to use `--function <name>` together with `--enforce-contract`
to set an explicit entry point. This way, ESBMC will **havoc** parameters and
global variables, verifying that the function satisfies its contract under all
possible input states — rather than exploring the entire `main` call chain.

```bash
esbmc main.c --enforce-contract increment --function increment
```
{{< /callout >}}

### Replace (Abstraction)

Used to substitute complex function calls with their contracts when verifying
larger programs.

| Command | Description |
|---------|-------------|
| `esbmc main.c --replace-call-with-contract <function_name>` | Replace calls to a specific function |
| `esbmc main.c --replace-all-contracts` | Replace calls to all `__ESBMC_contract` annotated functions |

When a function call is replaced:

1. ESBMC **asserts** that the precondition is met at the call site.
2. It **havocs** the targets specified by the `__ESBMC_assigns` clause (or all
   static globals if no assigns clause is present).
3. Finally, it **assumes** the post-condition holds.

{{< callout type="warning" >}}
Replace mode is an **over-approximation**: havocing introduces nondeterminism
that may not occur in the actual implementation. If ESBMC reports a
verification failure in replace mode, it does not necessarily indicate a real
bug — the counterexample trace may be infeasible under the concrete function
body. Currently, ESBMC does not provide a witness to confirm whether a
counterexample from contract abstraction corresponds to a genuine bug.

When a failure is reported, examine the counterexample trace to understand which
havoced values caused the violation, then refine the contract accordingly — for
example, by adding tighter postconditions in `__ESBMC_ensures` or narrowing the
modification frame in `__ESBMC_assigns` to reduce spurious nondeterminism.
{{< /callout >}}

### Combined Enforce and Replace

Both modes can be used together to verify one function while abstracting others:

```bash
esbmc main.c --enforce-contract func_a --replace-call-with-contract func_b
```

- `func_a`: Implementation verified against its contract
- `func_b`: Calls replaced with contract abstraction

{{< callout type="info" >}}
Automatically combining enforce and replace for all annotated
functions (i.e., using `--enforce-all-contracts` and `--replace-all-contracts`
together) is **not currently supported**. Enforcing and replacing every
function simultaneously provides little benefit over running ESBMC without
contracts at all. Instead, use the per-function options to selectively enforce
one function while replacing its callees.
{{< /callout >}}

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

### Memory Freshness Contract

This example demonstrates using `__ESBMC_is_fresh` to verify that a function
correctly allocates memory:

```c
#include <stdlib.h>

void create_array(int **arr, int n) {
    __ESBMC_requires(n > 0);
    __ESBMC_ensures(__ESBMC_is_fresh(*arr, n * sizeof(int)));
    __ESBMC_assigns(*arr);

    *arr = malloc(n * sizeof(int));
}

int main(void) {
    int *data;
    create_array(&data, 5);
    return 0;
}
```

Verify with: `esbmc main.c --enforce-contract create_array`

### Compositional Verification (X.509 Parsing)

This example demonstrates compositional verification on a simplified X.509
certificate parser. Each function is specified with its own contract, then
verified independently — trusting its callees' contracts rather than inlining
their bodies.

The code below is adapted from the
[ANSSI-FR/x509-parser](https://github.com/ANSSI-FR/x509-parser) project
(licensed under GPLv2). Contracts have been added for ESBMC verification.

```c
#include <stddef.h>

typedef unsigned char u8;
typedef unsigned int u32;

#define X509_FILE_LINE_NUM_ERR 1000
#define ERROR_TRACE_APPEND(x) do { } while(0)

typedef struct { int dummy; } cert_parsing_ctx;

#define CLASS_UNIVERSAL 0
#define ASN1_TYPE_SEQUENCE 16

/*
 * Parse ASN.1 tag + length.
 * On success: hdr_len == 2, hdr_len + data_len <= len.
 */
int parse_id_len(const u8 *buf, u32 len, int exp_class, u32 exp_type,
                 u32 *hdr_len, u32 *data_len)
{
    __ESBMC_requires(buf != NULL);
    __ESBMC_requires(hdr_len != NULL);
    __ESBMC_requires(data_len != NULL);
    __ESBMC_requires(len > 0);

    __ESBMC_assigns(*hdr_len, *data_len);

    __ESBMC_ensures((__ESBMC_return_value != 0) || (*hdr_len == 2));
    __ESBMC_ensures((__ESBMC_return_value != 0) || (*data_len <= 125));
    __ESBMC_ensures((__ESBMC_return_value != 0) || (*hdr_len + *data_len <= len));
    __ESBMC_ensures((__ESBMC_return_value != 0) || (*hdr_len + *data_len >= 2));
    __ESBMC_ensures((__ESBMC_return_value == 0) || (__ESBMC_return_value == -1));

    if (len < 2)
        return -1;
    if (buf[0] != 0x30)
        return -1;

    u8 len_byte = buf[1];
    if ((len_byte & 0x80) == 0) {
        *hdr_len = 2;
        *data_len = len_byte;
        if (*hdr_len + *data_len > len)
            return -1;
        return 0;
    }
    return -1;
}

/* Parse a full X.509 certificate blob. */
int parse_x509_cert(cert_parsing_ctx *ctx, const u8 *buf, u32 len)
{
    __ESBMC_requires(ctx != NULL);
    __ESBMC_requires(!(len > 0) || (buf != NULL));

    __ESBMC_assigns(ctx->dummy);

    if (buf == NULL || len == 0)
        return -1;

    ctx->dummy = 1;
    return 0;
}

/*
 * Top-level entry: parse one certificate and report how many bytes were
 * consumed via *eaten.
 */
int parse_x509_cert_relaxed(cert_parsing_ctx *ctx, const u8 *buf,
                            u32 len, u32 *eaten)
{
    __ESBMC_requires(!((len > 0) && (buf != NULL)) || (len <= 128));
    __ESBMC_requires(eaten != NULL);
    __ESBMC_requires(ctx != NULL);

    __ESBMC_assigns(*eaten, ctx->dummy);

    __ESBMC_ensures((__ESBMC_return_value != 0) || (*eaten > 0));
    __ESBMC_ensures((__ESBMC_return_value != 0) || (*eaten <= len));
    __ESBMC_ensures(__ESBMC_return_value <= 1);

    u32 hdr_len = 0, data_len = 0;
    int ret;

    if ((ctx == NULL) || (buf == NULL) || (len == 0) || (eaten == NULL)) {
        ret = -X509_FILE_LINE_NUM_ERR;
        ERROR_TRACE_APPEND(X509_FILE_LINE_NUM_ERR);
        goto out;
    }

    ret = parse_id_len(buf, len, CLASS_UNIVERSAL, ASN1_TYPE_SEQUENCE,
                       &hdr_len, &data_len);
    if (ret) {
        ret = 1;
        ERROR_TRACE_APPEND(X509_FILE_LINE_NUM_ERR);
        goto out;
    }

    *eaten = hdr_len + data_len;

    ret = parse_x509_cert(ctx, buf, hdr_len + data_len);
    if (ret) {
        ERROR_TRACE_APPEND(X509_FILE_LINE_NUM_ERR);
        goto out;
    }

    ret = 0;

out:
    return ret;
}

int main(void)
{
    cert_parsing_ctx ctx;
    u8 buffer[128];
    u32 len;
    u32 eaten;

    __ESBMC_assume(len > 0 && len <= 128);

    int ret = parse_x509_cert_relaxed(&ctx, buffer, len, &eaten);

    if (ret == 0) {
        __ESBMC_assert(eaten > 0, "eaten > 0 on success");
        __ESBMC_assert(eaten <= len, "eaten <= len on success");
    }
    __ESBMC_assert(ret <= 1, "result <= 1");

    return 0;
}
```

**Verification steps:**

Step 1 — Enforce each function's contract independently, replacing callees
with their contracts so that only one function body is analyzed at a time:

```bash
# Verify parse_id_len (no callees)
esbmc x509.c --function parse_id_len \
    --enforce-contract parse_id_len --no-align-check

# Verify parse_x509_cert (no callees)
esbmc x509.c --function parse_x509_cert \
    --enforce-contract parse_x509_cert --no-align-check

# Verify parse_x509_cert_relaxed (replace callees with contracts)
esbmc x509.c --function parse_x509_cert_relaxed \
    --enforce-contract parse_x509_cert_relaxed \
    --replace-call-with-contract "*" --no-align-check
```

Step 2 — Verify the system-level property in `main` using only contracts
(no function body is unrolled):

```bash
esbmc x509.c --replace-call-with-contract "*" --no-align-check
```

This compositional workflow allows each function to be verified in isolation
against its own contract. Callers rely on those contracts instead of
re-analyzing callee implementations. Note that this introduces an additional
proof obligation per function, but the cost is offset when the same function is
called from many sites — each call site avoids re-unrolling the callee's body.
