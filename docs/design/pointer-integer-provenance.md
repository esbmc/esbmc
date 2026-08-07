# Object recovery through integer arithmetic

ESBMC models a pointer as an (object, offset) pair. An integer holds no object,
so every `uintptr_t` round trip has to be re-associated with an object before the
resulting pointer can be dereferenced. `value_sett::get_value_set_rec` does that
for `add2t`/`sub2t`: one operand names objects, the other is treated as a byte
displacement, and each object is carried across with its offset updated.

Nothing does it for `mul2t`. A multiplied address falls through to the bottom of
`get_value_set_rec` and is recorded as `unknown`, so

```c
uintptr_t u = (uintptr_t)&s;
u *= 2;
u -= (uintptr_t)&s;   /* u == (uintptr_t)&s again */
*(int *)u = 3;
```

reports a spurious violation, and so do `regression/esbmc/github_426_2`,
`github_426_3` and `github_426_4` (esbmc/esbmc#6545). `github_426_2` only ever
multiplies an `offsetof`, but `offsetof` expands to `(size_t)&((S *)0)->y` and is
address-derived like any other address.

Reads take the noisy direction — a spurious counterexample. Writes did not: with
no target resolved the store landed on the fallback symbol, every object the
pointer could alias kept its old value, and asserting the write had not happened
was proved (esbmc/esbmc#6804). `dereference()` now raises `invalid pointer` for a
write that resolves no target, so the store is reported rather than dropped.

## Why the obvious fix is unsound

The tempting one-line change is to treat an operand whose value set contains
nothing but `unknown` as the integer side of the arithmetic, so the subtraction
above recovers `s` from its other operand. It does make all four tests pass, and
it survives the whole regression suite.

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

## What a real fix needs

Either of:

- **Provenance as a linear form.** Represent an address-derived value as a
  combination of object bases with coefficients, so `2*&s - &s` reduces to
  `1*&s` and recovers offset 0 exactly. This is a model extension, not a local
  change: every consumer of `object_mapt` would have to understand coefficients.

- **An address-range obligation at the dereference.** Recover the object as
  above, but make the recovery a proof obligation rather than an assumption:
  claim `(uintptr_t)p >= (uintptr_t)&obj && (uintptr_t)p < (uintptr_t)&obj +
  sizeof(obj)` wherever the object came from a discarded `unknown`. That
  discharges the round trip and rejects both witnesses above.

Until one of them lands, recovering object identity through a multiplicative term
is a known limitation and the three `github_426_*` tests stay `KNOWNBUG`.
