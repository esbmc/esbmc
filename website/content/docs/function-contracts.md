---
title: "Function Contracts"
date: 2026-04-08T00:00:00Z
draft: false
weight: 4
---

Function contracts let you state what a function requires from its callers and
what it promises to deliver — as machine-checkable annotations rather than
prose comments. ESBMC uses these annotations in two ways: to **verify** that a
function's implementation lives up to its promise, and to **replace** a
function call with its promise so that callers can be verified without
re-analyzing the function body.

## Your first contract

Suppose you have a function that increments a counter:

```c
int counter = 0;

void increment(void) {
    counter++;
}
```

You believe `counter` increases by exactly one each time. Turn that belief into
a contract:

```c
int counter = 0;

void increment(void) {
    __ESBMC_requires(counter >= 0);
    __ESBMC_ensures(counter == __ESBMC_old(counter) + 1);

    counter++;
}
```

Two new lines appeared:

- `__ESBMC_requires(cond)` — the **pre-condition**: what the caller must
  guarantee before calling this function.
- `__ESBMC_ensures(cond)` — the **post-condition**: what the function
  guarantees when it returns.

Now ask ESBMC to check whether the implementation satisfies the contract:

```bash
esbmc file.c --enforce-contract increment --function increment
```

The `--function increment` flag tells ESBMC to start verification at
`increment` instead of `main`. ESBMC will treat every parameter and every
global variable as an arbitrary value (subject to the `requires` constraint)
and then check that `ensures` holds after the body runs.

{{< callout type="info" >}}
Always pair `--enforce-contract func` with `--function func`. Without it,
ESBMC follows the call chain from `main`, which limits the range of inputs the
function is tested against.
{{< /callout >}}

### Referring to the return value

When a function returns a value, use `__ESBMC_return_value` inside `ensures`:

```c
int add_one(int n) {
    __ESBMC_requires(n < INT_MAX);
    __ESBMC_ensures(__ESBMC_return_value == n + 1);

    return n + 1;
}
```

### Capturing the value at entry: `__ESBMC_old`

The counter example above used `__ESBMC_old(counter)`. This captures the value
of an expression *at the moment the function was called* — before the body
runs. It is only meaningful inside `ensures`.

Without `__ESBMC_old`, writing `counter == counter + 1` would be a tautology
(the left and right `counter` both refer to the post-call value). With it, you
can say "the post-call value is one more than the pre-call value":

```c
__ESBMC_ensures(counter == __ESBMC_old(counter) + 1);
```

`__ESBMC_old` works on any expression: a global variable, a field of a struct,
or a value reachable through a pointer.

## What a function may modify: `__ESBMC_assigns`

Consider a function that is supposed to update one global and leave another
untouched:

```c
int value = 0;
int other = 100;

void modify_value(void) {
    __ESBMC_assigns(value);
    __ESBMC_ensures(value == __ESBMC_old(value) * 2);

    value = value * 2;
    /* other must not be touched */
}
```

`__ESBMC_assigns(value)` declares that `modify_value` is only allowed to
change `value`. It does two things depending on which mode ESBMC is running:

**In enforce mode** (`--enforce-contract`): ESBMC checks that the function
body actually only modifies the declared targets. If the body writes to `other`
without `other` being listed, ESBMC reports a verification failure.

**In replace mode** (`--replace-call-with-contract`): when ESBMC replaces a
call to `modify_value` with its contract, it *havocs* (assigns an arbitrary
value to) only the listed targets. `other` keeps its concrete value, giving
callers stronger guarantees.

### Assigns targets you can declare

| Syntax | What it covers |
|--------|---------------|
| `__ESBMC_assigns(x)` | A scalar variable or global |
| `__ESBMC_assigns(p->field)` | A field via pointer |
| `__ESBMC_assigns(*p)` | Whatever a pointer points to |
| `__ESBMC_assigns(arr[i])` | A single array element |
| `__ESBMC_assigns(x, y, z)` | Multiple targets (up to 5) |
| `__ESBMC_assigns()` or `__ESBMC_assigns(0)` | Nothing — declares a pure function |

### What happens when there is no `assigns` clause?

In **replace mode**, ESBMC conservatively havocs all static global variables
and any pointer parameters. This is sound but imprecise: callers will see all
globals as arbitrary values after the call, which may produce spurious failures.
Adding an `assigns` clause — even a coarse one — limits the disruption.

In **enforce mode**, assigns compliance checking is still active. ESBMC
snapshots all reachable globals before the call and asserts they are unchanged
after, except for those listed.

## Two modes: enforce and replace

### Enforce — "does the function keep its promise?"

```bash
esbmc file.c --enforce-contract <function> --function <function>
```

ESBMC builds a checking harness:

1. Allocates fresh, unconstrained values for all parameters and globals.
2. **Assumes** the `requires` clause (restricts the space of inputs).
3. Runs the function body symbolically.
4. **Asserts** the `ensures` clause and assigns compliance.

If every path through the body satisfies the postcondition and the assigns
frame, the result is `VERIFICATION SUCCESSFUL`. Otherwise, ESBMC reports a
counterexample showing which input values and which execution path caused a
violation.

### Replace — "trust the promise, skip the body"

```bash
esbmc file.c --replace-call-with-contract <function>
```

At each call site, ESBMC replaces the call with three steps:

1. **Asserts** the `requires` clause (the caller must have satisfied it).
2. **Havocs** the targets in `assigns` (models all possible side effects).
3. **Assumes** the `ensures` clause (takes the postcondition as given).

The function body is never unrolled. This keeps the verification of the caller
fast, at the cost of trusting the contract. If the contract is wrong, the
caller verification may produce a false result — which is why enforce mode
exists.

{{< callout type="warning" >}}
Replace mode is an over-approximation. Havocing introduces nondeterminism that
may not occur in the real implementation. A failure in replace mode does not
always correspond to a real bug — examine the counterexample to see whether the
havoced values are actually reachable under the concrete body. If they are not,
tighten the `ensures` clause or narrow the `assigns` targets.
{{< /callout >}}

### Using both together

The real power comes from combining the two modes. Enforce one function while
replacing its callees:

```bash
esbmc file.c --enforce-contract caller \
             --replace-call-with-contract callee \
             --function caller
```

`caller`'s body is verified using `callee`'s contract as a trusted
abstraction. Each function is verified in isolation.

## Compositional verification

The following TLV (Tag-Length-Value) parser has three levels of calls:

```
parse_message → parse_header → validate_tag
```

Each function is annotated with a contract. The verification proceeds
bottom-up: leaf functions first, then their callers, then `main`.

```c
typedef unsigned char u8;
typedef unsigned int  u32;

/* Leaf: check that a tag byte is in range */
int validate_tag(u8 tag)
{
    __ESBMC_ensures(__ESBMC_return_value == 0 || __ESBMC_return_value == -1);
    __ESBMC_ensures(__ESBMC_return_value != 0 || (tag >= 0x01 && tag <= 0x04));

    if (tag >= 0x01 && tag <= 0x04)
        return 0;
    return -1;
}

/* Mid: parse the 2-byte header */
int parse_header(const u8 *buf, u32 buf_len,
                 u32 *hdr_len, u32 *payload_len)
{
    __ESBMC_requires(buf != NULL && hdr_len != NULL && payload_len != NULL);
    __ESBMC_requires(buf_len > 0);
    __ESBMC_assigns(*hdr_len, *payload_len);
    __ESBMC_ensures(__ESBMC_return_value == 0 || __ESBMC_return_value == -1);
    __ESBMC_ensures(__ESBMC_return_value != 0 || *hdr_len == 2);
    __ESBMC_ensures(__ESBMC_return_value != 0 || *payload_len <= 125);
    __ESBMC_ensures(__ESBMC_return_value != 0 || *hdr_len + *payload_len <= buf_len);

    if (buf_len < 2) return -1;
    if (validate_tag(buf[0]) != 0) return -1;
    u8 len_byte = buf[1];
    if (len_byte & 0x80) return -1;
    *hdr_len    = 2;
    *payload_len = len_byte;
    if (*hdr_len + *payload_len > buf_len) return -1;
    return 0;
}

/* Top: parse a complete message */
int parse_message(const u8 *buf, u32 len, u32 *consumed)
{
    __ESBMC_requires(buf != NULL && consumed != NULL);
    __ESBMC_requires(len > 0 && len <= 128);
    __ESBMC_assigns(*consumed);
    __ESBMC_ensures(__ESBMC_return_value == 0 || __ESBMC_return_value == -1);
    __ESBMC_ensures(__ESBMC_return_value != 0 || (*consumed > 0 && *consumed <= len));

    u32 hdr = 0, payload = 0;
    if (parse_header(buf, len, &hdr, &payload) != 0) return -1;
    *consumed = hdr + payload;
    return 0;
}

int main(void) {
    u8 buffer[128];
    u32 len, consumed;
    __ESBMC_assume(len > 0 && len <= 128);

    int ret = parse_message(buffer, len, &consumed);
    if (ret == 0) {
        __ESBMC_assert(consumed > 0, "consumed > 0 on success");
        __ESBMC_assert(consumed <= len, "consumed <= len on success");
    }
    return 0;
}
```

**Step 1 — Verify each function in isolation.**

Start at the leaves. Each call uses `--replace-call-with-contract "*"` so that
only the target function's body is unrolled:

```bash
# No callees — verify the leaf directly
esbmc tlv.c --function validate_tag \
            --enforce-contract validate_tag

# parse_header calls validate_tag — replace it
esbmc tlv.c --function parse_header \
            --enforce-contract parse_header \
            --replace-call-with-contract validate_tag

# parse_message calls parse_header — replace it
esbmc tlv.c --function parse_message \
            --enforce-contract parse_message \
            --replace-call-with-contract parse_header
```

**Step 2 — Verify the system property using only contracts.**

Replace every annotated function so that `main` is verified without unrolling
any callee bodies:

```bash
esbmc tlv.c --replace-call-with-contract "*"
```

Each function is analyzed once, independently. The cost of re-unrolling the
same callee at every call site is eliminated.

## Loop contracts and the frame rule

Loop invariants tell ESBMC what remains true across every iteration. The
**loop frame rule** adds a complementary claim: which variables a loop is
allowed to change. Variables not listed are guaranteed to be untouched, and
ESBMC checks this.

```c
int main(void) {
    int i = 0;
    int j = 42;

    __ESBMC_loop_invariant(i >= 0 && i <= 10);
    __ESBMC_loop_assigns(i);
    while (i < 10)
        i++;

    /* j was not listed in loop_assigns — ESBMC can prove it is still 42 */
    __ESBMC_assert(j == 42, "j unchanged");
    return 0;
}
```

Run with:

```bash
esbmc file.c --loop-invariant-check --loop-frame-rule
```

`--loop-invariant-check` activates the loop invariant checker.
`--loop-frame-rule` additionally enforces the frame: after the loop havoc step,
every variable not listed in `__ESBMC_loop_assigns` is constrained to its
pre-loop value. Without this flag, the havoc step makes those variables
nondeterministic, and the assertion on `j` would fail.

The two macros must be placed **before the loop**, as statements ending with
`;`. Both may list up to five targets; use `__ESBMC_loop_assigns()` with no
arguments to declare that the loop modifies nothing.

{{< callout type="info" >}}
`--loop-frame-rule` requires `--loop-invariant-check`. It does not work with
`--loop-invariant` (the combined k-induction mode).
{{< /callout >}}

## Quantified conditions

For properties over all elements of an array, use `__ESBMC_forall` or
`__ESBMC_exists` inside `requires` or `ensures`:

```c
#define N 10

int find_min(int *a, int n)
{
    int i;
    __ESBMC_requires(a != NULL);
    __ESBMC_requires(n > 0 && n <= N);
    __ESBMC_requires(
        __ESBMC_forall(&i, !(i >= 0 && i < n) || (a[i] >= -100 && a[i] <= 100))
    );
    __ESBMC_ensures(__ESBMC_return_value >= -100);
    __ESBMC_ensures(__ESBMC_return_value <= 100);

    int m = a[0];
    for (int j = 1; j < n; j++)
        if (a[j] < m) m = a[j];
    return m;
}
```

The `__ESBMC_forall(&i, body)` reads: "for all values of `i`, `body` holds."
The variable must be declared in scope and passed by address.

{{< callout type="warning" >}}
`__ESBMC_forall` and `__ESBMC_exists` require a solver that supports
quantifiers. Use `--z3`:

```bash
esbmc file.c --enforce-contract find_min --function find_min --z3
```
{{< /callout >}}

## Memory freshness: `__ESBMC_is_fresh`

`__ESBMC_is_fresh(ptr, size)` asserts that `ptr` points to a valid, freshly
allocated block of at least `size` bytes that does not alias any existing
memory.

In `requires`: the caller must provide a freshly allocated pointer.
In `ensures`: the function promises to return a freshly allocated block.

```c
void create_buffer(char **out, int n) {
    __ESBMC_requires(n > 0);
    __ESBMC_ensures(__ESBMC_is_fresh(*out, n));

    *out = malloc(n);
}

void fill_buffer(char *buf, int n) {
    __ESBMC_requires(__ESBMC_is_fresh(buf, n));
    __ESBMC_requires(n > 0);
    __ESBMC_ensures(buf[0] == 'A');

    buf[0] = 'A';
}
```

## Bulk annotation with `__ESBMC_contract`

When many functions carry contracts, it is convenient to mark them once and
process them all together. The `__ESBMC_contract` attribute does this:

```c
__ESBMC_contract
void increment(void) {
    __ESBMC_requires(counter >= 0);
    __ESBMC_assigns(counter);
    __ESBMC_ensures(counter == __ESBMC_old(counter) + 1);
    counter++;
}

__ESBMC_contract
void reset(void) {
    __ESBMC_assigns(counter);
    __ESBMC_ensures(counter == 0);
    counter = 0;
}
```

Then enforce or replace all annotated functions in one command:

| Option | Effect |
|--------|--------|
| `--enforce-all-contracts` | Enforce every `__ESBMC_contract`-annotated function |
| `--replace-all-contracts` | Replace calls to every `__ESBMC_contract`-annotated function |

These options only affect explicitly annotated functions. Non-annotated
functions are left untouched.

**Difference from the `"*"` wildcard.** The `"*"` argument to
`--enforce-contract` and `--replace-call-with-contract` is broader: it matches
every function that has at least one contract clause, whether annotated or not.
`--enforce-all-contracts` and `--replace-all-contracts` match only annotated
functions.

## Quick reference

| Construct | Where | Purpose |
|-----------|-------|---------|
| `__ESBMC_requires(cond)` | Function body | Pre-condition |
| `__ESBMC_ensures(cond)` | Function body | Post-condition |
| `__ESBMC_assigns(t1, t2, ...)` | Function body | Modification frame |
| `__ESBMC_return_value` | Inside `ensures` | Return value of the function |
| `__ESBMC_old(expr)` | Inside `ensures` | Value of `expr` at function entry |
| `__ESBMC_is_fresh(ptr, size)` | `requires` / `ensures` | Memory freshness |
| `__ESBMC_forall(&var, body)` | `requires` / `ensures` | Universal quantifier (needs `--z3`) |
| `__ESBMC_exists(&var, body)` | `requires` / `ensures` | Existential quantifier (needs `--z3`) |
| `__ESBMC_loop_invariant(cond)` | Before loop | Loop invariant |
| `__ESBMC_loop_assigns(v1, ...)` | Before loop | Loop modification frame |
| `__ESBMC_contract` | Function attribute | Marks function for bulk processing |

| Option | Purpose |
|--------|---------|
| `--enforce-contract <f>` | Verify `f` against its contract |
| `--replace-call-with-contract <f>` | Replace calls to `f` with its contract |
| `--enforce-all-contracts` | Enforce all `__ESBMC_contract`-annotated functions |
| `--replace-all-contracts` | Replace calls to all `__ESBMC_contract`-annotated functions |
| `--loop-invariant-check` | Enable loop invariant checking |
| `--loop-frame-rule` | Enable loop frame rule (requires `--loop-invariant-check`) |

## Known limitations

The following cases are not yet fully supported. KNOWNBUG regression tests
document each one explicitly.

**Array assigns is bounded to 100 elements.** The nondet-witness approach used
for `__ESBMC_assigns(arr[i])` ranges over indices 0 to 99
(`ARRAY_ALLOC_ELEMS = 100`). If `i` can exceed 99, assigns compliance for
out-of-range writes will not be detected.

**Global array element assigns is unsupported.** `__ESBMC_assigns(global[i])`
does not work correctly for global arrays. Use `__ESBMC_assigns(global)` (the
whole array) as a conservative alternative.

**Multi-level pointer assigns is unsupported.** `__ESBMC_assigns(*p)` and
`__ESBMC_assigns(p->field)` work for a single level of indirection. Patterns
like `__ESBMC_assigns(p->sub->field)` are not classified and violations on the
untracked sub-fields will not be caught.

**Quantifiers require Z3.** `__ESBMC_forall` and `__ESBMC_exists` are not
supported by Boolector or other backends. Pass `--z3` when using quantified
conditions.
