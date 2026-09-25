# Object recovery through integer arithmetic

ESBMC models a pointer as an (object, offset) pair. An integer holds no object,
so every `uintptr_t` round trip has to be re-associated with an object before the
resulting pointer can be dereferenced. `value_sett::get_value_set_rec` does that
for `add2t`/`sub2t`: one operand names objects, the other is treated as a byte
displacement, and each object is carried across with its offset updated.

Nothing does it for `mul2t`. A multiplied address falls through to the bottom of
`get_value_set_rec` and is recorded as `unknown`, which is correct — `2*&s` does
not point into `s`. The defect was what happened next: an operand set holding
nothing but `unknown` counted as non-empty, so the *following* subtraction saw
two non-empty sets, hit the unhandled case, and dropped the whole expression to
`unknown` even though its other operand still named an object. So

```c
uintptr_t u = (uintptr_t)&s;
u *= 2;
u -= (uintptr_t)&s;   /* u == (uintptr_t)&s again */
*(int *)u = 3;
```

reported a spurious violation, as did `regression/esbmc/github_426_2`,
`github_426_3` and `github_426_4` (esbmc/esbmc#6545). `github_426_2` only ever
multiplies an `offsetof`, but `offsetof` expands to `(size_t)&((S *)0)->y` and is
address-derived like any other address.

Reads took the noisy direction — a spurious counterexample. Writes did not: with
no target resolved the store landed on the fallback symbol, every object the
pointer could alias kept its old value, and asserting the write did not happen
was proved (esbmc/esbmc#6804). So the gap missed bugs as well as inventing them,
and recovering the object fixes both directions: `regression/esbmc/github_6804`
is the write witness and is now `CORE`.

Failing every write that resolves no target is not the answer: the k-induction
inductive step havocs value sets on purpose, so such a check cannot tell a lost
store from a deliberate over-approximation and fires on every inductive step.

## Why narrowing the set is unsound

The tempting one-line change is to treat an operand whose value set contains
nothing but `unknown` as the integer side of the arithmetic, and to *drop* that
`unknown` from the result, so the subtraction above recovers `s` from its other
operand. It makes all four tests pass and survives the whole regression suite.

It is still wrong. `unknown` is the top element of the lattice — *may be any
object, including one we have not named* — not *is an integer*. Removing it makes
the points-to set exhaustive, and `dereference.cpp` emits the
`dereference failure: invalid pointer` property **only** for `unknown`/`invalid`
entries. Narrowing the set therefore deletes the property outright:

```c
char a[64], b[64];
unsigned long u = nondet_ulong();
char *p = (char *)(u * 8 - (uintptr_t)&a[0]);
if ((uintptr_t)p == (uintptr_t)&b[0]) {
  *p = 3;
  assert(b[0] == 0);      /* proved, on a path where p == &b[0] */
}
```

The branch is reachable, ESBMC's own model puts `p` at `&b[0]`, the write
executes — and the assertion is proved. The surviving bounds and alignment checks
apply to the fabricated object `a`, not to wherever `p` actually points, so they
catch nothing here. Trading a spurious counterexample for a false proof is the
wrong direction for a verifier.

`regression/esbmc/ptr_int_mul_unknown_alias` and `ptr_int_mul_unknown_wild` pin
both witnesses.

## Widening the set instead

The unsoundness above is caused by the *removal*, not by the recovery. Keeping
`unknown` in the result and adding the other operand's objects alongside it —
`{unknown}` becomes `{unknown, s}` rather than `{s}` — recovers the round trip
without deleting any property:

- On the round trip, `p` equals `&s`, the guard for `s` holds, and the write
  lands on `s.x`. The `unknown` arm's guard (`p` names no listed object) is
  unsatisfiable, so the proof goes through.
- On both witnesses above, `p` names neither listed object, the `unknown` arm is
  reachable, and `dereference failure: invalid pointer` fires exactly as before —
  same property, same line.

Each added object is a *guarded* alternative keyed on `POINTER_OBJECT(p)`, so a
pointer that does not name it cannot reach that arm. Adding objects to a may-set
can therefore only remove spurious counterexamples; it cannot hide a store,
because the `unknown` arm still catches every address outside the set.

This is what `value_sett::get_value_set_rec` now does for `add2t`/`sub2t` when
one operand's set holds nothing but `unknown`: the `unknown` is unioned into the
result first, and the other operand is then treated as the pointer side with a
nondet offset. The four `github_426_*` / `ptr_int_multiply_roundtrip` tests are
`CORE` as a result, and the two witnesses above stay `CORE` and still fail.

## What full provenance would still need

Object recovery is now a may-approximation, not an exact reconstruction: the
offset is nondet, so `2*&s - &s` is known to be *somewhere in* `s` rather than at
offset 0. Recovering the offset exactly needs one of:

- **Provenance as a linear form.** Represent an address-derived value as a
  combination of object bases with coefficients, so `2*&s - &s` reduces to
  `1*&s` and recovers offset 0 exactly. This is a model extension, not a local
  change: every consumer of `object_mapt` would have to understand coefficients.

- **An address-range obligation at the dereference.** Claim
  `(uintptr_t)p >= (uintptr_t)&obj && (uintptr_t)p < (uintptr_t)&obj +
  sizeof(obj)` wherever the object came from a discarded `unknown`, turning the
  recovery into a proof obligation rather than an assumption.

Neither is needed for the tests above, which the guarded may-set already
discharges.
