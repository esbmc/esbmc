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

`do`-`while` loops are verified by this mode. A `do`-`while` head is the first
instruction of the body rather than a guard, so the verification branch used to
copy an empty body and discharge the inductive step against no iteration at all
— every invariant passed, including a plainly wrong one, and a false post-loop
assertion was proved ([#7497](https://github.com/esbmc/esbmc/pull/7497)).

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

Three caveats:

- A claim after the loop is checked against the abstraction, not against the
  program, so a **correct but too weak invariant admits states the program
  cannot reach**. Such a claim is reported `UNKNOWN` rather than `FAILED`, with
  the reason attached — an over-approximation can prove a claim, never refute
  it:

  ```
  ** Results:
  main.c, function main
    PASSED   [main.assertion.1]  line 11  loop invariant base case
    PASSED   [main.assertion.2]  line 11  loop invariant inductive step
    UNKNOWN  [main.assertion.3]  line 16  assertion s == 3 (loop invariant too
             weak to prove this claim: the counterexample is against the havoc
             abstraction, not a reachable state of the program)

  ** 0 of 3 properties failed, 2 passed, 1 unknown
  WARNING: every violated claim lies downstream of a loop invariant havoc, so
  its counterexample is against the abstraction rather than the program;
  strengthen the invariant to decide the claim

  VERIFICATION UNKNOWN
  ```

  Strengthen the invariant until it entails the property.

  The downgrade is not unconditional. Before reporting `UNKNOWN`, ESBMC asks the
  solver whether the claim can hold at all on a feasible abstract path. If that
  probe is UNSAT, *no* abstract state satisfies the claim; the concrete states
  are a subset of the abstract ones, so the violation is real and the claim is
  reported `FAILED`. An invariant strong enough to pin the counterexample —
  `sn == (i - 1) * a` over an accumulator loop — therefore refutes a false
  post-loop assertion in this mode, which it could not do before
  ([#7626](https://github.com/esbmc/esbmc/pull/7626)). What is downgraded is the
  case the probe leaves open: the abstraction admits the claim holding *and*
  admits it failing, so the counterexample is the abstraction's and not the
  program's.

  What reports `FAILED` regardless of the probe is a claim *ahead* of every
  havoc, the invariant's own inductive step and its assigns-compliance check,
  and a loop the schema declined. An **outermost** loop's base case does too,
  since no havoc precedes it; an inner loop's base case sits inside the outer
  body, downstream of the outer havoc, so it is treated with everything else
  there.
- Storage a loop writes **through a dereference** has no symbol for the
  modified-variable analysis to name, so the pointee is havocked *through the
  pointer* and symex resolves it against its own value set. That covers a stack,
  heap or `__ESBMC_is_fresh` pointee alike, including one written by a callee
  ([#7518](https://github.com/esbmc/esbmc/pull/7518)). Before this, `(*p)--` in
  the body left `p`'s pointee at its pre-loop value, and an assertion about it
  after the loop was decided against state the loop had overwritten.
- Where even that leaves nothing to havoc, the schema **declines the loop**
  instead of claiming a proof. Such a loop is left to the unwinder, with a
  warning — `loop invariant at <location> not checked beyond its base case: the
  loop writes through a pointer the havoc cannot cover` — and its base case is
  still checked, since that runs from the concrete pre-loop state.
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

**Manual Invariant Specification:** Outside
[`--synthesise-loop-invariants`](#synthesising-invariants-for-affine-loops),
which covers one loop shape, users must write the invariants themselves. ESBMC
does not infer or validate an invariant before verification. An incorrect
invariant leads to a failed base-case assertion in `--loop-invariant` mode, or
an undecided claim in `--loop-invariant-check` mode.

> **Note:** `--loop-invariant-check` havocs every loop-modified variable, so an
> invariant that does not constrain them enough leaves the claims after the loop
> undecided (commonly an integer-overflow report), and they are reported
> `UNKNOWN`. `--loop-invariant` does not have this failure mode. If you hit it,
> either strengthen the invariant or, when the property follows from the
> invariant alone, switch to `--loop-invariant`.

**`--k-induction-parallel` still reports a downgraded claim as `FAILED`.** The
downgrade is recorded in the parallel driver but the verdict does not follow it
back across the fork
([#7516](https://github.com/esbmc/esbmc/issues/7516)).

## Mode Summary

|                          | Unrolls the loop | Exit reasoning | Weak invariant           |
| ------------------------ | ---------------- | -------------- | ------------------------ |
| `--loop-invariant`       | yes              | no             | falls back to unrolling  |
| `--loop-invariant-check` | no               | yes            | reports `UNKNOWN` unless the invariant refutes the claim |

Programs without loop invariant annotations continue to use the standard
k-induction unwinding approach under either flag.

## Synthesising Invariants for Affine Loops

```sh
esbmc file.c --synthesise-loop-invariants
```

k-induction cannot prove a property that needs a *relation* between a loop
counter and an accumulator. The interval domain is non-relational, so at the
loop head it knows the counter's range and nothing tying the accumulator to it.
`--synthesise-loop-invariants` recognises affine counter/accumulator loops and
emits the closed form as a `LOOP_INVARIANT`, which the existing
`--loop-invariant-check` schema then discharges.

```c
#include <stdint.h>
#include <assert.h>

int main(void) {
    uint64_t i = 1, sn = 0;
    uint32_t n;
    uint64_t a;
    __ESBMC_assume(n >= 1);

    while (i <= n) {
        sn = sn + a;
        i++;
    }

    assert(sn == (uint64_t)n * a);
    return 0;
}
```

The loop bound `n` is symbolic, so there is no `k` at which unwinding closes
this, and the inductive step has nothing relating `sn` to `i`.
`--k-induction` runs for over 300 s without reaching a verdict.
`--synthesise-loop-invariants` derives `sn == (i - 1) * a` and finishes in under
a second:

```
Synthesised loop invariants for 1 loop
...
** 0 of 3 properties failed, 3 passed
Solver: Bitwuzla 0.9.1 • Decision procedure total time: 0.007s
VERIFICATION SUCCESSFUL
```

At this width only Bitwuzla — ESBMC's default — discharges the exit obligation,
which is a multiplier-equivalence miter; the program above is pinned as
`regression/bitwuzla/synth_loop_invariant_sum64`, and
`regression/esbmc/synth_loop_invariant_sum` carries the same shape at narrower
widths so that every platform runs it.

**The synthesised invariant is checked, not assumed.** It goes through the same
assert / havoc-assume / assert schema a hand-written one does, so a wrong
candidate fails a claim rather than producing an unsound proof. The failure mode
of synthesis is a spurious failure or a useless invariant, never a false proof.

**It trades bug-finding for proving.** Cutting the loop removes the bounded
traces a plain BMC run would search. Measured over
`regression/{esbmc,esbmc-unix,k-induction,loop-invariants}`, synthesis fires on
87 of 2716 files, of which 9 (10%) lose a bug that bounded BMC finds; no file
gains a false proof or a false alarm. It is opt-in and off by default for that
reason.

The flag implies `--loop-invariant-check`, `--check-vacuity`, and
`--multi-property` unless a k-induction phase is selected. Because
`--check-vacuity` applies to the whole run rather than only to loops the
synthesis reached, a program with no loop at all can report `UNKNOWN` where it
reported `SUCCESSFUL`; `--no-vacuity-check` turns that back off.

### What it recognises, and what it declines

The recognised shape is a loop whose head is `IF !(i <op> B) GOTO exit` with a
straight-line body containing a unit-step counter `i = i + 1` and an accumulator
`s = s + e` whose addend `e` is free of loop-modified variables. From those it
emits the accumulator's closed form and a two- or three-disjunct counter bound.

Two regimes, and the restrictions follow from the cost of discharging the exit
obligation rather than from the recogniser:

| Addend | Counter | Why |
| --- | --- | --- |
| symbolic | unsigned, entering at 0 or 1 | The bound must stay at two disjuncts: a third arm leaves the solver proving two 64-bit multiplier circuits equivalent, which does not terminate. Two disjuncts discharge in ~1s; three do not finish in 120s. |
| literal | signed or unsigned, any literal entry | With no symbolic multiplier there is nothing to miter, so the extra arms are free and establishment is unconditional. |

Declined outright:

- **Any program that can reach `__ESBMC_spawn_thread`.** Cutting a loop deletes
  its interleaving points, so a claim only another thread could violate is no
  longer reachable in the cut program and would be reported passed.
- **A loop the user has already annotated**, and synthesis anywhere inside a
  function a user invariant's expression calls, transitively — otherwise the
  user's own marker reads a havoc-abstracted return value.
- **Every loop under `--unsigned-overflow-check`**, and a signed loop whose
  weaker bound is observable (a body that asserts, or signed overflow checking
  on). The synthesised guards are themselves instrumented by `goto_check`, so a
  closed form emitted at a watched type would draw overflow claims on arithmetic
  the user never wrote.

Decrementing loops and comparisons other than `<` / `<=` are not recognised. The
restrictions and the design constraint they follow from are documented in
`src/goto-programs/goto_invariant_synthesis.h`.

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
