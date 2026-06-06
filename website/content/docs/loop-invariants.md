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
`__ESBMC_loop_invariant(condition)`, placed within the loop body.

## Verification Modes

ESBMC provides **two distinct modes** for loop invariant verification.

### `--loop-invariant` — Combined Mode (Default, Recommended)

This is the recommended mode. It integrates invariant checking with k-induction
for robust analysis, eliminating the spurious counterexamples that the legacy
mode could produce for correct-but-weak invariants.

> **Note:** `--loop-invariant` now implicitly enables k-induction. No extra
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

### `--loop-invariant-check` — Legacy Mode

The original three-part havoc abstraction is preserved under this flag for
backward compatibility. It replaces the annotated loop with:

1. **Base-case assertion** — invariant holds on entry
2. **Havoc + Assume** — abstracts the loop body nondeterministically
3. **Inductive-step assertion** — invariant still holds after one iteration

This mode avoids loop unrolling entirely, making it faster when the only goal is
checking invariant inductivity without k-induction overhead. However, it **may
produce spurious counterexamples** for invariants that are correct but weak,
because the havoc step can assign values outside the expected program state
without proper constraint propagation.

> Use `--loop-invariant-check` only when speed is the priority and you
> understand the false-positive risk.

## Example: Basic Loop Invariant Usage

```c
int main() {
    int i = 0;
    int sum = 0;

    __ESBMC_loop_invariant(i >= 0 && i <= 1000 && sum == i * 10);
    while (i < 1000) {
        sum += 10;
        i++;
    }

    assert(sum == 10000);  // Successfully verified
    return 0;
}
```

Verify with:

```bash
esbmc file.c --loop-invariant
```

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

> **Note:** The integer overflow false-positive issue present in the legacy mode
> (caused by unconstrained havoc operations) has been resolved in the new
> `--loop-invariant` combined mode. If you observe such false positives, ensure
> you are not using `--loop-invariant-check`.

## Backward Compatibility

The `--loop-invariant` flag now routes to the new combined mode. The legacy
behavior is accessible via `--loop-invariant-check`. Programs without loop
invariant annotations continue to use the standard k-induction unwinding
approach.

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
