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

Until it lands, the three `github_426_*` tests stay `KNOWNBUG` as
under-approximations of a soundness defect, not as a stated limitation. Tracked
as esbmc/esbmc#6804.
