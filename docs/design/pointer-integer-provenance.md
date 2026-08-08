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

## The same gap also proves false assertions

An earlier revision of this document called the spurious counterexample "the
noisy direction — a spurious counterexample, not a missed bug." **That was
wrong** (esbmc/esbmc#6804). The write that goes missing from `s` is not reported
anywhere, so asserting that it *did not* happen is proved:

```c
uintptr_t u = (uintptr_t)&s;
u *= 2;
u -= (uintptr_t)&s;   /* u == (uintptr_t)&s */
int *p = (int *)u;
*p = 3;               /* really writes s.x */
assert(s.x == 42);    /* false in C; ESBMC reports SUCCESSFUL */
```

`VERIFICATION SUCCESSFUL` under both Bitwuzla and Z3, while the same program
compiled with gcc aborts on the assertion at `-O0` and `-O2`. The additive
round trip (`u += 8; u -= 8`) reports `FAILED` on the identical assertion, so
the verdict turns on which operator was used, not on the program's semantics.

The mechanism is that a top value set silently swallows the write:

1. `get_value_set_rec` records `unknown` (`value_set.cpp:776-784`).
2. `dereference()` sets `known_exhaustive = false` and builds a fallback
   `make_failed_symbol(type)` (`dereference.cpp:564-570`).
3. The `unknown` entry reaches `deref_invalid_ptr` and returns nil, so it
   contributes no real target (`dereference.cpp:793-797`, `579-580`).
4. In write mode the resulting lvalue is therefore the failed symbol. The write
   lands on a fresh unconstrained symbol and no real object is havoc'd.

The `invalid pointer` property is not a net here. It is guarded by a
solver-decided `invalid_pointer(deref_expr)` (`dereference.cpp:951`, `968-977`),
which is false exactly when the address *is* recoverable — the case where the
lost write matters. It fires only when the pointer is genuinely wild.

This is not specific to multiplication. A plain nondet integer cast to a pointer
and constrained to alias an object loses the write the same way; there it
happens to be masked by the alignment check, and `--no-pointer-check` exposes the
same false proof. Multiplication is simply the case where the recovered address
is too well-formed for any co-firing pointer check to mask it.

Note that the solver layer is not at fault. `convert_typecast_to_ptr`
(`smt_casts.cpp:435-509`) already resolves an integer to its object by
address-range comparison against `__ESBMC_ptr_obj_start_N`/`_end_N`, and the
counterexample for `github_426_3` shows it recovering `p == &s` exactly. The
static value-set layer discards that before the solver is consulted.

## Why the obvious value-set fix is unsound

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
catch nothing here.

`regression/esbmc/ptr_int_mul_unknown_alias` and `ptr_int_mul_unknown_wild` pin
both witnesses. Both use `nondet_ulong()`, so `p` can be wild and the
`invalid pointer` property fires; that is why they stay green while the lost
write described above sits underneath them untested.
`regression/esbmc/ptr_int_mul_lost_write` pins the missed-bug direction.

## What a real fix needs

Recovering provenance in the value set — whether as a linear form over object
bases, or as a speculative recovery discharged by an address-range obligation at
the dereference — addresses only the multiplicative case. Step 4 above mishandles
a top value set whatever produced it, so neither closes the hole.

The fix that does is to stop discarding the case split. When
`known_exhaustive` is false, `dereference()` should build the same
`same_object(p, &obj)`-guarded chain it already builds for value-set entries, but
over the address-space objects, keeping the failed symbol only as the final
`else`. That recovers `s` for the round trip above, sends the write in
`ptr_int_mul_unknown_alias` to `b` where it belongs, and still falls through to
`invalid pointer` for `ptr_int_mul_unknown_wild`, which names no object at all.
It needs no `mul2t` case, no coefficients in `object_mapt`, and no new proof
obligation: widening a points-to set can only add cases to the split, never
delete a property.

The open question is cost — the split is over every object in the address space,
so this may need to ship behind a flag before it can be default-on.

## What landed: `--deref-unknown-objects`

`dereferencet::widen_to_known_objects()` implements the fix above. On a write
through a non-exhaustive value set it appends every object in
`value_sett::object_numbering` to the points-to set, each with an unknown offset,
and lets the existing loop build the `same_object`-guarded chain over them; the
failed symbol stays as the final `else`. Objects are appended in object-number
order so the chain — and hence the formula — does not depend on the numbering's
hash order. The `unknown` entry is never removed, so `deref_invalid_ptr` still
fires; liveness, dynamic invalidation and const-ness are left to `valid_check`,
which guards each failure by `same_object` exactly as it does for entries the
value set supplied.

It closes both directions of the defect at once:

| test | default | `--deref-unknown-objects` | correct |
|---|---|---|---|
| `ptr_int_mul_lost_write` | SUCCESSFUL | FAILED | FAILED |
| `ptr_int_multiply_roundtrip` | FAILED | SUCCESSFUL | SUCCESSFUL |
| `github_426_2`, `github_426_4` | FAILED | SUCCESSFUL | SUCCESSFUL |
| `github_2512_11` | FAILED | SUCCESSFUL | SUCCESSFUL |
| `ptr_int_mul_unknown_wild` | FAILED | FAILED | FAILED |

`github_426_3` still reports FAILED, so the trio is not fully discharged.

`github_2512_11` deserves note: it writes `e.d = 3` through an `offsetof`-derived
pointer and then asserts `e.d`, and gcc holds the assertion at `-O0` and `-O2`.
Its `CORE` expectation of `VERIFICATION FAILED` therefore pins the lost write
rather than a real bug — a fourth test attributable to this defect, and the only
one enforcing the wrong answer rather than merely being parked as `KNOWNBUG`. It
has to be corrected when the flag goes default-on.

### Why it is opt-in

Instrumenting `dereference()` over the core C suite (1659 tests) shows the write
path is rare but the split is wide: 32 tests reach a write through a
non-exhaustive value set, over 491 sites, and `object_numbering` at those sites
holds a median of 21 objects, 86 at the 90th percentile and 104 at most. That
fan-out is enough to push several tests from seconds to over the timeout —
`github_1807` 2s → >150s, `31_cdaudio` 6s → >150s — which is why the flag
defaults off. Narrowing the split to type-compatible, still-live objects is the
obvious next step; the reads that dominate the count (11640 sites) need no split
at all, since a failed symbol over-approximates a read soundly.

Until the flag can be default-on, the three `github_426_*` tests stay `KNOWNBUG`
as under-approximations of a soundness defect, not as a stated limitation.
Tracked as esbmc/esbmc#6804.
