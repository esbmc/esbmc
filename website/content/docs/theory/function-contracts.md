---
title: Function Contracts and Modular Verification
weight: 35
---

Contracts let you specify *what* a function guarantees instead of re-verifying
*how* it is implemented at every call site. ESBMC uses contracts for **modular
(assume-guarantee) verification**: a function is checked once against its
contract, and its calls are then replaced by the contract — turning a
whole-program proof into a set of smaller, independent ones.

## Specifying a contract

A function is marked for contract processing with the `__ESBMC_contract` macro,
and its clauses are written with intrinsics in the body. These are always
declared, so an annotated file still compiles in plain BMC mode (where the
clauses are dropped as no-ops) and only become active under the contract flags
below:

```c
#include <limits.h>

__ESBMC_contract
int abs_val(int x)
{
  __ESBMC_requires(x > INT_MIN);              // precondition (−INT_MIN overflows)
  __ESBMC_ensures(__ESBMC_return_value >= 0); // postcondition
  __ESBMC_assigns();                          // frame: modifies nothing
  return x < 0 ? -x : x;
}
```

The contract vocabulary:

| Intrinsic | Meaning |
|---|---|
| `__ESBMC_requires(cond)` | Precondition assumed on entry / asserted at calls |
| `__ESBMC_ensures(cond)` | Postcondition guaranteed on return |
| `__ESBMC_return_value` | The function's return value, for use in `ensures` |
| `__ESBMC_old(x)` | The pre-state value of `x`, for relating output to input |
| `__ESBMC_assigns(a, b, …)` | Frame condition: the only locations the function may modify |

## Two modes

**Enforce** a contract — prove the implementation satisfies it:

```sh
esbmc file.c --enforce-contract abs_val
```

ESBMC assumes `requires`, runs the body, and asserts `ensures` and the
`assigns` frame on every path. `--enforce-all-contracts` enforces every
annotated function.

**Replace** calls with a contract — use it as a summary at call sites:

```sh
esbmc file.c --replace-call-with-contract abs_val
```

Each call to `abs_val` is replaced by: assert its `requires`, havoc the
locations in its `assigns`, and assume its `ensures`. The callee's body is not
re-explored, so a caller is verified against the *specification* of its callees.
Combining the two modes — enforce each function once, replace it everywhere it
is called — gives a modular proof whose cost does not blow up with call depth.

## Loop contracts

The same idea applies to loops, replacing unbounded unwinding with an inductive
argument:

```c
__ESBMC_loop_invariant(i >= 0 && i <= n);
__ESBMC_loop_assigns(i, sum);
for (i = 0; i < n; i++) sum += a[i];
```

`__ESBMC_loop_invariant` states a property preserved by every iteration;
`__ESBMC_loop_assigns` declares which locations the loop may change, so
everything else is known unchanged across it. Loop invariants are applied with
`--loop-invariant` (a combined loop-invariant + k-induction mode), and the
`assigns` frame rule with `--loop-frame-rule` (which requires
`--loop-invariant-check`). A valid invariant lets ESBMC summarise the loop
instead of unrolling it, which — like
[k-induction](/docs/theory/verification-algorithms#k-induction) — can prove
programs whose loop bounds are not statically fixed.
