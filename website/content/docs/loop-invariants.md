---
title: Loop Invariants
weight: 6
---

```sh
esbmc file.c --loop-invariant
```

ESBMC supports user-provided loop invariants as an alternative to expensive loop
unwinding. This is particularly beneficial for programs with large loop bounds
or unbounded loops, where traditional k-induction may become computationally
prohibitive or hit iteration limits.

Like [function contracts](/docs/function-contracts), loop invariants are
expressed with built-in constructs placed in the source. The core construct is
`__ESBMC_loop_invariant(condition)`, placed **before the loop** as a statement
ending with `;`.

## Choosing a Mode

ESBMC provides **two distinct modes**, and picking the wrong one is the most
common reason a correct invariant appears not to help. The deciding question is
**what your post-loop property needs in order to follow**:

| Your property follows from…                                       | Use                      | Cost in the loop bound         |
| ----------------------------------------------------------------- | ------------------------ | ------------------------------ |
| the invariant **alone**                                            | `--loop-invariant`       | independent — closes at `k = 2` |
| the invariant **together with the negated loop condition**         | `--loop-invariant-check` | independent — loop is cut      |

The second row is easy to miss. An invariant like `i <= N && sum == i * 10`
only yields `sum == N * 10` once you also know `!(i < N)`, i.e. `i == N` at
exit. That final step is *exit reasoning*, and only `--loop-invariant-check`
performs it. Under `--loop-invariant` such a property falls back to bounded
unrolling and will report `VERIFICATION UNKNOWN` once the bound exceeds the
k-induction step limit.

Properties over an array filled by the loop — `__ESBMC_forall(&i, !(i < N) ||
p[i] == 0)` and friends — are almost always in the second row, since they need
`i == N` at exit.

## Verification Modes

### `--loop-invariant` — Combined Mode

Integrates invariant checking with k-induction. The invariant is used as an
assumption that strengthens the inductive step, so whenever the property
follows from the invariant alone, verification closes at a small `k`
regardless of how large the loop bound is.

> **Note:** `--loop-invariant` implicitly enables k-induction. No extra
> flags are required.

**How it works — two-branch transformation:**

```
IF !nondet_bool() GOTO loop_head       // Non-deterministically skip to Branch 2

// --- Branch 1: Inductivity Check ---
ASSERT(INV)                            // Base case: invariant holds on entry
HAVOC(loop_vars)
ASSUME(INV)
ASSUME(loop_entry_cond)
ASSERT(INV)                            // Inductive step: invariant still holds
ASSUME(false)                          // Terminate Branch 1

loop_head:
// --- Branch 2: K-Induction ---
ASSUME(INV)                            // Use invariant as a hint for k-induction
GOTO loop_head
```

**Expected outcomes:**

| Invariant Quality     | Branch 1 Result                                      | K-Induction Result                    |
| --------------------- | ---------------------------------------------------- | ------------------------------------- |
| Wrong (not inductive) | ASSERT fails — clear "invariant not inductive" error | —                                     |
| Correct but weak      | Passes                                               | Proves property via forward condition |
| Correct and strong    | Passes                                               | Closes at inductive step              |

The last row holds when the property follows from the invariant alone. If it
also needs the negated loop condition, the inductive step cannot close and
k-induction falls back to unrolling — see [Choosing a Mode](#choosing-a-mode).

### `--loop-invariant-check` — Havoc Abstraction Mode

Applies the classic Hoare rule, replacing the annotated loop with:

1. **Base-case assertion** — invariant holds on entry
2. **Havoc + Assume** — abstracts the loop body nondeterministically
3. **Inductive-step assertion** — invariant still holds after one iteration

The loop is then cut, so this mode **avoids loop unrolling entirely** and its
cost does not grow with the loop bound. It is the only mode that performs exit
reasoning (`invariant && !condition` at the loop exit), which makes it the
required choice for the second row of [Choosing a Mode](#choosing-a-mode).

Two caveats:

- It **may produce spurious counterexamples** for invariants that are correct
  but too weak, because the havoc step can assign values outside the expected
  program state without proper constraint propagation. Strengthen the invariant
  until it entails the property you are proving.
- Cutting the loop establishes **partial correctness** only: termination is not
  proved. Use `--termination` separately if you need it.

## Example: Property Follows From the Invariant Alone

`x == y` is exactly what the assertion needs, so the inductive step closes at
`k = 2` and the loop bound is irrelevant — this verifies as quickly at `100000`
as at `10`.

```c
#include <assert.h>

int main(void) {
    unsigned int x = 0;
    unsigned int y = 0;

    __ESBMC_loop_invariant(x == y);
    while (x < 100000) {
        x++;
        y++;
    }

    assert(x == y);
    return 0;
}
```

```bash
esbmc file.c --loop-invariant
# VERIFICATION SUCCESSFUL — Solution found by the inductive step (k = 2)
```

## Example: Property Needs the Exit Condition

Here `sum == 10000` follows only from `sum == i * 10` *and* `i == 1000`, and
the latter needs `!(i < 1000)` at exit. This is the case that requires
`--loop-invariant-check`; under `--loop-invariant` it reports
`VERIFICATION UNKNOWN` because the bound exceeds the k-induction step limit.

```c
#include <assert.h>

int main(void) {
    unsigned int i = 0;
    unsigned int sum = 0;

    __ESBMC_loop_invariant(i <= 1000 && sum == i * 10);
    while (i < 1000) {
        sum += 10;
        i++;
    }

    assert(sum == 10000);
    return 0;
}
```

```bash
esbmc file.c --loop-invariant-check
# VERIFICATION SUCCESSFUL
```

The same shape appears whenever a loop fills an array and the postcondition
quantifies over it, which is why array contracts under `--enforce-contract`
normally want `--loop-invariant-check`:

```c
__ESBMC_ensures(__ESBMC_forall(&i, !(i < N) || (a->e[i] >= 0 && a->e[i] < Q)));

__ESBMC_loop_invariant(i <= N && __ESBMC_forall(&j, !(j < i) ||
                       (a->e[j] >= 0 && a->e[j] < Q)));
for (i = 0; i < N; i++)
  a->e[i] = reduce(a->e[i]);
```

With `--loop-invariant-check` this discharges in a fraction of a second for any
`N`, and `--unwind` only has to cover the rest of the function rather than the
loop.

## Companion Options

The following options can be combined with the k-induction proof rule to produce
or strengthen inductive invariants:

- `--interval-analysis` — Enable interval analysis for integer variables and
  inject assume statements into the program.
- `--add-symex-value-sets` — Enable value-set analysis for pointers and inject
  assume statements.
- `--loop-invariant` — Use user-provided loop invariants with the combined
  k-induction mode (described above).

## Known Limitations

**Nested Loop Support:** The current implementation does not correctly handle
nested loops with multiple invariants. State management between inner and outer
loops requires further refinement.

**Manual Invariant Specification:** Users must manually specify correct loop
invariants. ESBMC will not infer or validate invariants before verification. An
incorrect invariant will lead to a failed base-case assertion in
`--loop-invariant` mode, or potentially a spurious result in
`--loop-invariant-check` mode.

> **Note:** `--loop-invariant-check` havocs every loop-modified variable, so an
> invariant that does not constrain them enough can yield a false positive
> (commonly an integer-overflow report). `--loop-invariant` does not have this
> failure mode. If you hit it, either strengthen the invariant or, when the
> property follows from the invariant alone, switch to `--loop-invariant`.

## Mode Summary

|                          | Unrolls the loop | Exit reasoning | Weak invariant           |
| ------------------------ | ---------------- | -------------- | ------------------------ |
| `--loop-invariant`       | yes              | no             | falls back to unrolling  |
| `--loop-invariant-check` | no               | yes            | may report false positive |

Programs without loop invariant annotations continue to use the standard
k-induction unwinding approach under either flag.

## Loop Frame Rule (`--loop-frame-rule`)

A loop invariant says what stays true across iterations. The **loop frame rule**
adds the complementary claim: which variables the loop is allowed to change.
Variables not listed in `__ESBMC_loop_assigns` are guaranteed to be untouched —
and ESBMC checks this.

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

Without `--loop-frame-rule`, the havoc step makes every loop-modified variable
nondeterministic, so the assertion on `j` would fail despite `j` never being
touched. With the flag, ESBMC snapshots all variables not in
`__ESBMC_loop_assigns` before the havoc and assumes they are unchanged
afterward.

Both macros must be placed **before the loop**, as statements ending with `;`.
`__ESBMC_loop_assigns` supports up to five targets; use `__ESBMC_loop_assigns()`
with no arguments to declare that the loop modifies nothing.

> `--loop-frame-rule` requires `--loop-invariant-check`. It does not work with
> `--loop-invariant` (the combined k-induction mode).
