---
title: "Function Contracts"
date: 2026-04-08T00:00:00Z
draft: false
weight: 5
---

Function contracts let you state what a function requires from its callers and
what it promises to deliver — as machine-checkable annotations rather than prose
comments. ESBMC uses these annotations in two ways: to **verify** that a
function's implementation lives up to its promise, and to **replace** a function
call with its promise so that callers can be verified without re-analyzing the
function body.

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
- `__ESBMC_ensures(cond)` — the **post-condition**: what the function guarantees
  when it returns.

Now ask ESBMC to check whether the implementation satisfies the contract:

```bash
esbmc file.c --enforce-contract increment --function increment
```

The `--function increment` flag tells ESBMC to start verification at `increment`
instead of `main`. ESBMC will treat every parameter and every global variable as
an arbitrary value (subject to the `requires` constraint) and then check that
`ensures` holds after the body runs.

{{< callout type="info" >}} Always pair `--enforce-contract func` with
`--function func`. Without it, ESBMC follows the call chain from `main`, which
limits the range of inputs the function is tested against. {{< /callout >}}

### Referring to the return value

When a function returns a value, use `__ESBMC_return_value` inside `ensures`:

```c
#include <limits.h>

int add_one(int n) {
    __ESBMC_requires(n < INT_MAX);
    __ESBMC_ensures(__ESBMC_return_value == n + 1);

    return n + 1;
}
```

### Capturing the value at entry: `__ESBMC_old`

The counter example above used `__ESBMC_old(counter)`. This captures the value
of an expression _at the moment the function was called_ — before the body runs.
It is only meaningful inside `ensures`.

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

`__ESBMC_assigns(value)` declares that `modify_value` is only allowed to change
`value`. It does two things depending on which mode ESBMC is running:

**In enforce mode** (`--enforce-contract`): ESBMC checks that the function body
actually only modifies the declared targets. If the body writes to `other`
without `other` being listed, ESBMC reports a verification failure.

**In replace mode** (`--replace-call-with-contract`): when ESBMC replaces a call
to `modify_value` with its contract, it _havocs_ (assigns an arbitrary value to)
only the listed targets. `other` keeps its concrete value, giving callers
stronger guarantees.

### Assigns targets you can declare

| Syntax                                      | What it covers                     |
| ------------------------------------------- | ---------------------------------- |
| `__ESBMC_assigns(x)`                        | A scalar variable or global        |
| `__ESBMC_assigns(p->field)`                 | A field via pointer                |
| `__ESBMC_assigns(*p)`                       | Whatever a pointer points to       |
| `__ESBMC_assigns(arr[i])`                   | A single array element             |
| `__ESBMC_assigns(x, y, z)`                  | Multiple targets (up to 5)         |
| `__ESBMC_assigns()` or `__ESBMC_assigns(0)` | Nothing — declares a pure function |

### What happens when there is no `assigns` clause?

In **replace mode**, ESBMC conservatively havocs all static global variables and
any pointer parameters. This is sound but imprecise: callers will see all
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

Step 1 is unconstrained in the extent of pointer parameters too, **in
entry-harness mode only**. That is, when the enforced function is also the
`--function` entry point, so ESBMC has to invent the arguments. Under a plain
`--enforce-contract f` the pointer parameters come from the real caller and
nothing below applies.

In entry-harness mode a pointer parameter is backed by an object whose size is
nondeterministic, so the body may only dereference it as far as the `requires`
clause justifies. State the extent with
[`__ESBMC_is_fresh`](#memory-freshness-__esbmc_is_fresh):

```c
void f(int *p) {
    __ESBMC_requires(__ESBMC_is_fresh(p, 21 * sizeof(int)));
    p[20] = 1;   // in bounds: the contract says p addresses 21 ints
}
```

Without that clause, `p != NULL` alone says nothing about how many elements `p`
addresses, so `p[20]` is reported as an out-of-bounds write. ESBMC emits a
warning naming any pointer parameter whose extent the contract leaves unstated.

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
fast, at the cost of trusting the contract. If the contract is wrong, the caller
verification may produce a false result — which is why enforce mode exists.

{{< callout type="warning" >}} Replace mode is an over-approximation. Havocing
introduces nondeterminism that may not occur in the real implementation. A
failure in replace mode does not always correspond to a real bug — examine the
counterexample to see whether the havoced values are actually reachable under
the concrete body. If they are not, tighten the `ensures` clause or narrow the
`assigns` targets. {{< /callout >}}

### Using both together

The real power comes from combining the two modes. Enforce one function while
replacing its callees:

```bash
esbmc file.c --enforce-contract caller \
             --replace-call-with-contract callee \
             --function caller
```

`caller`'s body is verified using `callee`'s contract as a trusted abstraction.
Each function is verified in isolation.

## Compositional verification

The following TLV (Tag-Length-Value) parser has three levels of calls:

```
parse_message → parse_header → validate_tag
```

Each function is annotated with a contract. The verification proceeds bottom-up:
leaf functions first, then their callers, then `main`.

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

Each function is analyzed once, independently. The cost of re-unrolling the same
callee at every call site is eliminated.

## Loop contracts and the frame rule

`__ESBMC_loop_assigns` and `--loop-frame-rule` extend loop invariants with a
frame claim — which variables a loop may change — and check that all others are
untouched. These are documented in the
[Loop Invariant Support](/docs/loop-invariants) section.

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

The `__ESBMC_forall(&i, body)` reads: "for all values of `i`, `body` holds." The
variable must be declared in scope and passed by address.

{{< callout type="warning" >}} `__ESBMC_forall` and `__ESBMC_exists` require a
solver that supports quantifiers. Use `--z3`:

```bash
esbmc file.c --enforce-contract find_min --function find_min --z3
```

{{< /callout >}}

## Memory freshness: `__ESBMC_is_fresh`

`__ESBMC_is_fresh(ptr, size)` asserts that `ptr` points to a valid, freshly
allocated block of at least `size` bytes that does not alias any existing
memory.

In `requires`: the caller must provide a freshly allocated pointer. In
`ensures`: the function promises to return a freshly allocated block.

Because a fresh block aliases nothing, a `requires`-side `__ESBMC_is_fresh` is
also how a contract states that a pointer parameter is separate from the
others. Enforcement grants it — the parameter gets its own allocation, and is
excluded from the aliasing described under "Pointer parameters may alias"
below. Replacement therefore checks it, asserting at the call that the argument
shares no object with any other pointer argument, and that the object it points
into really does extend `size` bytes past it. Both halves are obligations on
the caller, so a program that passed a smaller object than the contract asked
for now reports a violated `requires` where it previously verified.

Two consequences worth knowing:

- Only an unconditionally asserted `__ESBMC_is_fresh` states separation. Under
  a guard, as in `__ESBMC_requires(n <= 0 || __ESBMC_is_fresh(p, n))`, the
  contract claims nothing on the other branch, so neither is anything granted
  to the callee nor demanded of the caller.
- This is stronger than CBMC's `__CPROVER_is_fresh`, which separates a pointer
  only from other fresh pointers in the same contract. A ported contract that
  marks one parameter fresh and leaves another as a plain pointer will be
  rejected here if a caller passes the same object to both.

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

| Option                    | Effect                                                       |
| ------------------------- | ------------------------------------------------------------ |
| `--enforce-all-contracts` | Enforce every `__ESBMC_contract`-annotated function          |
| `--replace-all-contracts` | Replace calls to every `__ESBMC_contract`-annotated function |

These options only affect explicitly annotated functions. Non-annotated
functions are left untouched.

**Difference from the `"*"` wildcard.** The `"*"` argument to
`--enforce-contract` and `--replace-call-with-contract` is broader: it matches
every function that has at least one contract clause, whether annotated or not.
`--enforce-all-contracts` and `--replace-all-contracts` match only annotated
functions.

## Contracts in Python

The Python frontend lowers `__ESBMC_requires` / `__ESBMC_ensures` clauses too,
so `--enforce-contract` and `--replace-call-with-contract` work on Python
functions:

```python
def double(x: int) -> int:
    __ESBMC_requires(x > 0)
    __ESBMC_ensures(__ESBMC_return_value > x)
    return 2 * x
```

```bash
esbmc main.py --enforce-contract double
```

`__ESBMC_return_value` takes its type from the function's return annotation.
Clauses are inert under a plain BMC run, exactly as in C.

A clause the lowering cannot express is rejected with a diagnostic naming the
clause and line — a function call or subscript inside the condition, a
reference whose type cannot be determined, `__ESBMC_return_value` inside
`requires` or on a function annotated `-> None` — rather than being silently
dropped. `__ESBMC_old`, `__ESBMC_assigns`, and the quantified forms are not yet
available in Python.

## Quick reference

| Construct                       | Where                  | Purpose                               |
| ------------------------------- | ---------------------- | ------------------------------------- |
| `__ESBMC_requires(cond)`        | Function body          | Pre-condition                         |
| `__ESBMC_ensures(cond)`         | Function body          | Post-condition                        |
| `__ESBMC_assigns(t1, t2, ...)`  | Function body          | Modification frame                    |
| `__ESBMC_return_value`          | Inside `ensures`       | Return value of the function          |
| `__ESBMC_old(expr)`             | Inside `ensures`       | Value of `expr` at function entry     |
| `__ESBMC_is_fresh(ptr, size)`   | `requires` / `ensures` | Memory freshness                      |
| `__ESBMC_forall(&var, body)`    | `requires` / `ensures` | Universal quantifier (needs `--z3`)   |
| `__ESBMC_exists(&var, body)`    | `requires` / `ensures` | Existential quantifier (needs `--z3`) |
| `__ESBMC_loop_invariant(cond)`  | Before loop            | Loop invariant                        |
| `__ESBMC_loop_assigns(v1, ...)` | Before loop            | Loop modification frame               |
| `__ESBMC_contract`              | Function attribute     | Marks function for bulk processing    |

| Option                             | Purpose                                                                              |
| ---------------------------------- | ------------------------------------------------------------------------------------ |
| `--enforce-contract <f>`           | Verify `f` against its contract                                                      |
| `--replace-call-with-contract <f>` | Replace calls to `f` with its contract                                               |
| `--enforce-all-contracts`          | Enforce all `__ESBMC_contract`-annotated functions                                   |
| `--replace-all-contracts`          | Replace calls to all `__ESBMC_contract`-annotated functions                          |
| `--loop-invariant-check`           | Enable loop invariant checking (see [Loop Invariant Support](/docs/loop-invariants)) |
| `--loop-frame-rule`                | Enable loop frame rule (requires `--loop-invariant-check`)                           |

A name these flags cannot act on is an error, not a silent no-op:

```
ERROR: --enforce-contract: cannot use 'typoo': no function of that name,
or the name is ambiguous
```

The same applies to a function with no body, a function with no contract
clause, a comma-separated list where one name is expected, and
`--enforce-all-contracts` / `--replace-all-contracts` when nothing matches. A
typo used to report `VERIFICATION SUCCESSFUL` for a run that applied no contract
at all.

`__ESBMC_is_fresh(EXPR, n)` likewise rejects a non-pointer `EXPR` rather than
lowering into an incoherent allocation.

## Python contracts

Contracts work on Python functions too. The clauses are written as ordinary
calls at the top of the function body, with no import and no declaration:

```python
def double(x: int) -> int:
    __ESBMC_requires(x > 0)
    __ESBMC_ensures(__ESBMC_return_value > x)
    return 2 * x
```

```
$ esbmc double.py --enforce-contract double
VERIFICATION SUCCESSFUL
```

Both modes work, and the diagnostics name the clause that failed:

```
Violated property:
  contract ensures
  return_value > x
```

Methods are addressed by their bare name, or by a full symbol ID when the name
is ambiguous:

```python
class Counter:
    def add(self, k: int) -> int:
        __ESBMC_requires(k > 0)
        __ESBMC_ensures(__ESBMC_return_value > 0)
        return k
```

```
$ esbmc counter.py --enforce-contract add
$ esbmc counter.py --enforce-contract "py:counter.py@C@Counter@F@add"
```

### No `__ESBMC_old` for parameters

Python passes scalars by value, so a parameter named in `ensures` already
denotes its value at entry, even when the body reassigns it:

```python
def bump(x: int) -> int:
    __ESBMC_requires(x > 0)
    __ESBMC_ensures(__ESBMC_return_value == x + 1)   # x is the entry value
    x = x + 1
    return x
```

`__ESBMC_old` is therefore unnecessary for parameters, and is not yet supported
in Python. It remains necessary in C, and will be needed in Python once
contracts reach mutable state.

### A clause must be a pure expression

A clause is lowered into a single assumption or assertion, so it has to survive
as one expression. Three constructs do not, and each is rejected with the line
and the reason rather than verified as something else:

| Written | Rejected because |
| --- | --- |
| `__ESBMC_requires(len(l) > 0)` | a call binds to a temporary local to the body, leaving the clause over a free symbol |
| `__ESBMC_ensures(__ESBMC_return_value == l[0])` | a subscript expands into a bounds-check branch, collapsing the clause to a constant |
| `__ESBMC_requires(o > 0)` where `o` has no inferable type | an untyped parameter is a `void *`, so the comparison is on a pointer |

```
ERROR: __ESBMC_requires clause at line 2 contains a function call; a contract
clause must be a pure expression
```

The call restriction is the one users hit first, and it is a general contract
defect rather than a Python one
([#6941](https://github.com/esbmc/esbmc/issues/6941)): the same clause in C is
silently mis-verified instead of rejected.

Note that the restriction is on the **clause**, not on the signature. A `list`
or `str` parameter is fine as long as the clause itself is pure:

```python
def take(l: list, n: int) -> int:
    __ESBMC_requires(n > 0)                          # accepted
    __ESBMC_ensures(__ESBMC_return_value == n)
    return n
```

An unannotated parameter is likewise fine when the frontend infers its type
from a call site; it is rejected only when the type really could not be
determined.

### What Python supports today

| Construct | Status |
| --- | --- |
| `__ESBMC_requires` / `__ESBMC_ensures` | supported, several of each per function |
| `__ESBMC_return_value` | supported, typed from the return annotation |
| `and`, `or`, `not` in a clause | supported |
| `--enforce-contract` / `--replace-call-with-contract` | supported |
| parameters annotated `int`, `float`, `bool` | supported |
| `list`, `str`, class instances, unannotated parameters | supported when the clause does not mention them |
| methods | supported |
| globals in a clause | supported |
| `__ESBMC_old` | not supported, and not needed for parameters (above) |
| `__ESBMC_assigns` | not supported |
| `__ESBMC_is_fresh` | not supported |
| `__ESBMC_forall` / `__ESBMC_exists` | not supported |
| `--enforce-all-contracts` / `--replace-all-contracts` | not supported, no Python equivalent of the `__ESBMC_contract` attribute |

Every unsupported construct reports what it is rather than failing obscurely:

```
ERROR: __ESBMC_assigns at line 5 is not supported in Python contracts yet
```

Each construct that is rejected has a `FUTURE` regression test stating the
verdict it should produce once implemented, so the test flips from passing to
"consider reclassifying" the moment support lands.
`--enforce-all-contracts` has no such test: it needs a Python surface for the
`__ESBMC_contract` marker, and until that is chosen there is no source file
whose verdict would change.

Without a contract flag the clauses state nothing and are dropped, so an
annotated file verifies exactly as it did before it was annotated.

Progress and the remaining work are tracked in
[#6938](https://github.com/esbmc/esbmc/issues/6938).

## Known limitations

The following cases are not yet fully supported. KNOWNBUG regression tests
document each one explicitly.

**The array-assigns witness index falls back to 100 elements when the extent is
unknown.** For `__ESBMC_assigns(arr[i])` the nondet witness index is clamped to
the array's real extent, `n / sizeof(elem)` when the pointer came from
`__ESBMC_is_fresh(a, n)`. Where no extent is recorded, such as a global pointer
or a run without `--function`, the bound falls back to
`WITNESS_IDX_FALLBACK_ELEMS = 100`, which can over-bound the index and report a
spurious bounds violation on a smaller array. The bound is a clamp rather than
an assumption: assuming the index range would force the extent to be at least
one element, and for a zero extent it would be an assumption of false, which
discharges the whole wrapper vacuously.

**Pointer parameters with no stated extent are not checked against the assigns
clause.** Proving that `*p` is unchanged means reading `*p` in the harness, and
against an unstated extent that read is itself out of bounds. Such parameters
are skipped rather than reported, so a contract that wants frame checking for a
pointer must state its extent with `__ESBMC_is_fresh`. The warning names the
parameters this applies to.

**Struct pointer parameters assume one element.** A `struct S *` parameter is
backed by a single stack-allocated `S`, so `s->field` is accepted even when the
contract states no extent for `s`. This is the same unstated assumption that
nondet extents remove for other pointer types, just narrowed to one element.
Moving struct parameters onto the same nondet-extent allocation is blocked on
[#6483](https://github.com/esbmc/esbmc/issues/6483): a heap-backed struct
parameter silently discharges `__ESBMC_old`-based `ensures` clauses.

**Pointer parameters may alias, and not every aliasing is explored.** A
contract that needs two pointer parameters to be separate has to say so, with
`__ESBMC_requires(p != q)` or `__ESBMC_is_fresh`; enforcement otherwise lets
them refer to the same object, because nothing in the contract rules it out and
nothing checks it at a call site. The aliasing offered is not exhaustive: it
relates parameters of identical pointer type only, so differently typed
parameters, `void *` and byte buffers are still assumed separate, as are
interior aliases such as `f(&s->a, &s->b)` or `f(buf, buf + 1)`, and aliasing
between a parameter and a global. A contract relying on any of those keeps an
assumption that is granted but never discharged.

**`__ESBMC_is_fresh` on anything but a bare parameter is not discharged at a
call.** `__ESBMC_is_fresh(g, n)` for a global, or `__ESBMC_is_fresh(s->p, n)`,
is honoured when enforcing — the lvalue gets its own allocation — but
replacement keys the separation obligation on a parameter position and emits
nothing for these forms.

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
