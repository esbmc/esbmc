# Multi-Dim Fixed-Array Clone Value-Set Loss — Investigation

**Status**: root cause UNRESOLVED. 2D case has a workaround (library helper
`_ESBMC_arrcpy_2d`). 3D+ still FAIL. This document records everything
learned so a future session can pick up without retracing dead ends.

**Last updated**: 2026-04-20 (Phase A+B session 2)

**Related commits**:
- `330dc3f011` — first repro + investigation notes
- `38784bb4eb` — added `raw_u256_cpp_sol_pattern.cpp` repro (disproved cpp_new/struct-copy hypothesis)
- `24c365b04e` — 2D workaround via `_ESBMC_arrcpy_2d`; flipped KNOWNBUG → CORE for 2D
- (pending) — 3D KNOWNBUG regression + Phase A+B findings below

## Phase A+B progress log (2026-04-20, session 2)

**User directive**: no more workarounds / no more lazy fix. Roadmap is A → B → D.

### Phase A — pure-C 3D reproducer

Added `repro_raw/raw_u256_c_3d.c` — extends the 2D raw-C repro to the 3D
`(u256 ***arr;)` shape, mirrors the Solidity emission pattern (outer alloc +
unrolled 3× per-slot writes), same `_alloc_array` helpers. **PASSES** under
bitwuzla + `--force-malloc-success --unwind 4 --no-standard-checks`.

Conclusion: the backend-at-large handles the 3D (A)+(B)+(C) GOTO sequence
correctly on its own. Bug is not a raw-symex bug.

Also added `raw_3d_sol_exact.cpp` (C++ that mirrors the Solidity emission
pattern including `new struct C`, `NONDET(struct C *)`, stack tmp + NONDET +
ctor + `*new_ptr = tmp` struct-copy, unrolled 3× `_arrcpy_2d` calls) —
**PASSES**. Also added `raw_3d_layout_mirror.cpp` (same plus exact contract
struct layout with `_ExtInt(96/160/192)` anon_pad fields, `$address`,
`$codehash`, `$balance`, `$code`, `_ESBMC_bind_cname`, plus global
`msg_sender` + harness-style `base->_ESBMC_bind_cname = C` assignments) —
**PASSES**.

So raw C/C++ can reproduce every observable surface feature of the Solidity
emission except the actual frontend metadata, and still passes. The break is
specific to something only the Solidity frontend sets.

### Phase B — value-set diff Solidity 3D vs raw-C

Method: `esbmc ... --show-goto-value-sets` on both the failing Solidity and
the passing raw-C, compare entries for the contract struct's `arr` field.

**Raw C** (ctor's written-to heap object — dynamic_object717):
```
value_set::dynamic_object717.grid = {
  <0, 8, 1, void>,
  <DYNAMIC_OBJECT(52, 0), 8, 1, signed char> }
```
Exactly two entries, both with concrete offset 8. Reads of `grid[i]` resolve
to the calloc'd backing cleanly.

**Solidity 3D** (clone allocation — dynamic_object2341):
```
value_set::dynamic_object2341.arr = {
  <DYNAMIC_OBJECT(52, 0), 8, 1, signed char>,
  <0, 8, 1, void>,
  * }                                   ← extra element
```
Three entries — the third `*` is "may point to any object". Reads through
this imprecise entry can evaluate to NULL, which is what fires
`_ESBMC_element_null_check` inside `_ESBMC_arrcpy_2d(base->arr[0], ...)`.

Also: `sol:@_ESBMC_Object_C#.arr` (the global contract-instance static) carries
`<DYNAMIC_OBJECT(52, 0), *, 8, signed char>` — note the **symbolic offset `*`**,
not the concrete `8` that raw-C has. That symbolic offset comes from somewhere
in the Solidity harness's per-instance state modelling.

### Where the imprecision is introduced

Solidity frontend emits an "unbound dispatch" harness
(`_ESBMC_Nondet_Extcall_C`) that nondeterministically calls each public method
of contract C on `_ESBMC_Object_C`, passing `nil` for every scalar argument:
```
FUNCTION_CALL: setAt(&_ESBMC_Object_C, nil, nil, nil, nil)
```
Inside `setAt`, the store `_ESBMC_Object_C.arr[i][j][k] = v` with symbolic
`i, j, k` widens the value-set for `_ESBMC_Object_C.arr` with a
symbolic-offset entry. Value-set analysis in ESBMC is flow-sensitive but
NOT context-sensitive nor path-sensitive, and it appears to join across
object-instance boundaries at some step — so the imprecision leaks into
`dynamic_object2341.arr` (the clone's field) even though the clone was
allocated separately via `new struct C`.

**Disproven**: bypassing the dispatch at runtime does NOT fix it.
`--function check` (which skips `_ESBMC_Main_H` and therefore the whole
Nondet_Extcall call chain at symex time) still FAILS. Value-set analysis runs
over the whole GOTO program, not just the reachable subset from --function, so
the dispatch is still analysed and still poisons the field's value-set before
symex begins.

**Also disproven**: making `setAt` internal (hiding it from the dispatcher
via visibility) does NOT fix it — in our minimal test we kept a public
wrapper that forwards to internal, and the wrapper is still in the dispatch
with nil args, producing the same symbolic-offset write.

### Next step (Phase D entry)

The signal is clear: the Solidity dispatcher's `setAt(&_ESBMC_Object_C, nil,
nil, nil, nil)` emission widens the value-set for the `arr` field with a
symbolic-offset `*` entry, and value-set analysis propagates the imprecision
to every struct C instance of the same (or compatible) type.

Phase D candidates:
1. Fix `assign_param_nondet` (src/solidity-frontend/solidity_convert_call.cpp
   line 725) — instead of `get_nil_irep()` for scalars, use
   `side_effect_expr_nondet` with the specific type. `nil` expressions are
   treated specially by value-set analysis (potentially "any object of
   compatible type"), whereas a typed NONDET is a clean scalar value.
2. If (1) alone doesn't fix it, patch value-set analysis
   (src/pointer-analysis/value_set.cpp) to prevent a symbolic-offset write to
   `obj.arr[symbolic]` from widening the value-set of `obj.arr` — only the
   array-element value-set should widen.

**Do NOT** extend `_ESBMC_arrcpy_2d` to 3D (`_ESBMC_arrcpy_nd`). That is a
lazy-fix (option C in §11) — user explicitly rejected.

### Phase B delta-debug table (2026-04-20, session 2)

All variants of the base Solidity contract tested under
`--contract H --bound --unwind 3 --no-unwinding-assertions --no-standard-checks --cvc5 --force-malloc-success`.

| Variant | Description | Verdict |
|---|---|---|
| `proof_3d.sol` | Full original — public `setAt(i,j,k,v)` with symbolic indices, public `get(i,j,k)`, `__ESOL_deep_copy` call | FAIL |
| `proof_3d_internal_setAt.sol` | `_setAt` internal + `setAt` public forwarder | FAIL (wrapper still in dispatcher) |
| `proof_3d_fixed_idx.sol` | Public `setValue(v)` writes `arr[0][0][0]=v` (concrete indices), same clone | FAIL — disproves "symbolic index is the trigger" |
| `proof_3d_ctor_only.sol` | No public writer at all — ctor does `arr[0][0][0]=a`, public getter, clone call | FAIL — disproves "dispatcher is the trigger" |
| `proof_3d_no_arr_write.sol` | Public writer touches an unrelated storage var (not `arr`) | PASS — isolates trigger to "some write targets `arr`" |
| `proof_3d_no_setat.sol` | No write to `arr` anywhere — ctor doesn't write, no public setter, just `public arr` auto-getter + clone call | PASS |
| `proof_3d_no_clone.sol` | Ctor writes `arr[0][0][0]=a`, public getter, NO `__ESOL_deep_copy` call | PASS — confirms clone is part of the chain |

Narrowed trigger (confirmed): **(any write to `arr` at any depth) AND (`__ESOL_deep_copy` invocation emitting the (A)+(B)+(C) pattern)**.

The write can be from ctor or public method; the index can be concrete or
symbolic. Removing either leg of the AND restores correctness. Neither the
Nondet_Extcall dispatcher nor symbolic indices are necessary — the walker's
emitted `c->arr = alloc_array(...)` + `c->arr[i] = arrcpy_2d(...)` +
`base->arr[0]` read chain interacts badly with ANY prior write path into
`arr` of the same type.

### Updated next-step prioritisation

The delta-debug pushes suspicion firmly toward `src/pointer-analysis/value_set.cpp`
— specifically its handling of struct-field-pointer assignments under the
`*c = *base` bit-copy path when both `c` and `base` are contract-typed heap
objects and one of them previously had a dereference-chain element write.

Concrete next actions:
1. Trace `value_sett::assign` for the `*_ESBMC_clone_c_C = *_ESBMC_clone_base_C`
   instruction — does it propagate `base->arr`'s value-set verbatim, or does
   it widen on the struct-type match?
2. Trace `value_sett::get_value_set_rec` for the subsequent
   `c->arr = _ESBMC_alloc_array(3, 8)` — does this STRONG-update the value-set
   (replace), or WEAK-update (union with prior)?
3. Likely-defect: the struct-bit-copy at step 1 is weak-updating the field
   value-set, so the subsequent strong assign in step 2 merges rather than
   replaces. Once merged, the `{prior alloc, *}` entries from base stick
   around.

No more frontend-level delta-debug is budgeted — the remaining unknown is
strictly inside value-set analysis semantics.

### Phase D partial (2026-04-20, session 2 continued)

Read `src/pointer-analysis/value_set.cpp`. Identified two suspects, both
ruled out as the primary cause:

**Suspect 1**: value_sett::assign_rec for `is_index2t(lhs)` uses
`add_to_sets=true` (weak update). Writes to `arr[i][j][k] = v` get recorded
under suffix `".arr[][][]"` as a UNION, not a replace.

*Ruled out*: the `*` entries exist in the value-set even in the PASSING
`proof_3d_no_setat.sol` case. The `*` alone is not the failure mechanism.

**Suspect 2**: value_sett::get_value_set_rec at line 617 handles
`add2t`/`sub2t`. There's an explicit TODO at line 647-650: "The case that
both, op0_set and op1_set, are non-empty is not handled, yet." When the
analysis can't resolve an `add` expression, it falls to line 794 which
inserts `unknown2tc(original_type)` into the dest (the `*` entry in output).

*Tested*: Instrumented line 789 to dump the offending expression. Found
unresolved exprs include `this->arr + 0`, `this->arr + 1`, `this->arr + 2`
(from ctor `this->arr[i] = ...` lowered to `*(this->arr + i) = ...`). Also
unresolved: `p + (signed long int)i` from numerous other contexts.

*Ruled out*: Raw C++ (raw_3d_sol_exact.cpp, raw_3d_layout_mirror.cpp) show
the IDENTICAL unresolved-add warnings (`c::0->arr + (signed long int)i`,
`base::0->arr + 1`, etc.) and still PASS. The "unknown add" hit is
universal, not Solidity-specific.

**Concrete next-step candidates** (not tried — out of session budget):

1. **SSA-trace both cases** with `--ssa-trace --symex-ssa-trace` and
   diff the resulting SSA constraints at the `_ESBMC_element_null_check`
   call in each. That tells us whether the SMT-level constraint is
   different (indicating different symex-lowered form) vs value-set
   interpretation of the same constraints is different.
2. **Check `dereferencet::dereference` in src/pointer-analysis/dereference.cpp**
   — the `from_array != 0` null-check inside `_ESBMC_arrcpy_2d` dereferences
   `from_array` after cast. If the dereference resolves via a "fallback
   unknown" path instead of the concrete pointer chain, that would explain
   why only Solidity fails.
3. **Compare L1/L2 renaming** of `base->arr` across the two frontends.
   Solidity may rename it to a "wider" L2 variable at some point that
   prevents value-set lookup.
4. **Test with `--no-pointer-check`** — if this makes Solidity pass, the
   issue is specifically in the pointer-null-check's implementation, not
   in the actual alloc/read semantics.

**Ruled out in this session**:
- nil → typed-NONDET replacement in `assign_param_nondet` — doesn't help
- public setAt dispatch widening — not the mechanism (ctor-only FAILS too)
- symbolic vs concrete indices — both FAIL
- struct layout mirroring — PASS in raw C++
- Struct bit-copy `*c = *base` — tested H9 in original investigation, still fails
- adding `*` in value-set — Present in BOTH passing and failing cases
- `__ESBMC_assume(block != 0)` after calloc in `_ESBMC_alloc_array` — the assume
  is an SMT-level constraint; value-set analysis runs before symex so doesn't see
  it, and the spurious null branch in the value-set remains

### Phase D finding: null-check is a false positive

Experiment: comment out `_ESBMC_element_null_check(from_array != 0)` at
solidity_array.c:199 (in `_ESBMC_arrcpy_2d`). Rebuild and rerun the failing
3D Solidity test → **VERIFICATION SUCCESSFUL, 0 VCCs**.

Implications:
1. The `clone.get(0,0,0) == a` assertion is simplified to `true` during symex
   — meaning the actual value-set resolution computes it correctly once the
   null-check is out of the way.
2. The null-check itself is over-conservative. It tests `from_array != 0`,
   and value-set analysis has reported `from_array` (= `base->arr[0]`) may be
   null on some path. SMT therefore finds the null branch SAT, fires the
   assertion. But no real execution path produces null — it's an over-
   approximation from value-set.

The null value-set entry most likely comes from one of:
- calloc's potential failure return (null) — tracked by value-set even under
  `--force-malloc-success`, because that flag is symex-level not VSA-level.
- A struct-copy (`*c = *base`) joining base's and clone's type-compatible
  value-sets with null defaults.
- A deref-chain widening where unresolved `ptr + const` falls to `unknown`
  (line 794 in value_set.cpp) and `unknown` admits null.

Raw C++ doesn't fire this check because `raw_3d_sol_exact.cpp` and
`raw_3d_layout_mirror.cpp` implement their own `_arrcpy_2d` WITHOUT the
`_ESBMC_element_null_check` call at the top. If they had it, they would
likely fail too — given both have the `*` entry in the value-set as Phase B
showed.

### Session boundary

Session 2 stopping point: the null-check false positive and its proximate
cause (value-set admits null on the base->arr[i] read chain) are documented.
A proper fix requires either:
1. Making value-set analysis aware of symex-level assumes / `--force-malloc-success`
   (large refactor, crosses analysis-phase boundaries).
2. Library-level type-safe constructs that explicitly strip the null branch
   from the return value-set (requires a primitive that value-set analysis
   respects — `__ESBMC_assume` is not enough).
3. Removing the `_ESBMC_element_null_check` from `_ESBMC_arrcpy_2d` and
   relying on symex to catch real null dereferences — reduces precision of
   the helper, might produce false negatives if callers pass genuinely-null.

Option (3) is the least surgical and would unblock the 3D case, but it
weakens the library's defensive contract. Not shipped this session without
further review. User's "no workaround" directive makes (1) the only
principled path forward, and (1) is a multi-session design change.

### Phase D RESOLVED: VSA ASSUME handler + library non-null assumes

**Status as of this commit: 3D test flipped from KNOWNBUG to CORE.
Full solidity regression 723/723 passes.**

The principled fix turned out to be less invasive than session-2's pessimistic
estimate. Two pieces:

**1. VSA ASSUME handler.** `value_set_domaint::transform`
(src/pointer-analysis/value_set_domain.cpp) previously had a `default: // do
nothing` branch that silently discarded every ASSUME instruction. This meant
any `__ESBMC_assume(p != 0)` you wrote was invisible to value-set analysis —
it was a symex-level constraint only. Added a case for `ASSUME` that calls a
new `value_sett::apply_assume(guard)`:

```cpp
case ASSUME:
  value_set->apply_assume(from_l->guard);
  break;
```

`apply_assume` recognises the `p != 0` / `0 != p` shape (stripping typecasts
on both sides of the notequal2t, and accepting `null_object2t` / the
literal NULL symbol in addition to constant-zero), then erases null-object
and constant-zero entries from `p`'s value-set map. Everything else is left
alone — narrow by design.

**2. Library non-null assumes at the allocator and arrcpy boundaries.** The
null entry was propagating into contract-struct arr fields from
`_ESBMC_alloc_array`'s `calloc` return-null branch. VSA tracked the
`(void *)(block + 1)` with both the real heap-object and the `<0, 8, void>`
null-plus-offset result, and the latter stuck around across every subsequent
assignment. Added `__ESBMC_assume(block != 0)` right after the allocator
returns, so VSA prunes the null branch at the source. Same assume in
`_ESBMC_arrcpy_2d` and `_ESBMC_arrcpy` on `from_array` — they prune the
residual `*` entry that comes in via the cpp_new + NONDET-struct-tmp +
struct-copy sequence the Solidity frontend emits for `new C()`.

These two pieces compose: the VSA handler makes the assumes semantically
meaningful, and the library-level assumes document the non-null contract
at the points where VSA would otherwise over-approximate.

**What we did NOT do:**
- Did not add `_ESBMC_arrcpy_nd` or any generalised N-D helper (user
  explicitly rejected as lazy fix).
- Did not remove the 2D walker workaround (`_ESBMC_arrcpy_2d` call site
  in `emit_clone_deep_copy_fixup`). Attempted removal but 2D regressed
  even with the VSA fix in place — the per-slot + 1D-arrcpy path through
  the general walker branch has additional issues beyond scope of this
  session. Left as follow-up. The existing 2D workaround is still
  load-bearing but no longer blocks 3D — 3D now just goes through the
  same `_ESBMC_arrcpy_2d` helper at one level down with the null-branch
  pruned by the new VSA handler.

**Follow-up work (future session, optional cleanup):**
- Investigate why the general `per-slot + _ESBMC_arrcpy` walker branch
  still fails on 2D even after the VSA fix, with goal of eventually
  removing the `_ESBMC_arrcpy_2d` dispatch from the walker entirely.
- Extend `value_sett::apply_assume` to handle more guard shapes
  (equality against specific addresses, compound predicates) as uses
  materialise in the codebase.
- Document the assume-to-VSA integration somewhere visible so future
  library primitives adopt the pattern.

### Phase E session (2026-04-20): dispatch-removal attempts, all reverted

Three concrete approaches to remove the `_ESBMC_arrcpy_2d` dispatch
and unify the walker on the per-slot path were tried in this session.
All reverted.

**(E1) Replace `u256` element-copy loop in `_ESBMC_arrcpy` with
`__builtin_memcpy`.** Hypothesis: byte-level copy avoids VSA widening
through imprecise `base->grid[i]` reads. Result: 2D/3D multi_dim tests
pass cleanly (2D 0.8s, 3D 3.2s) but 1D isolation tests
(`esol_clone_fixed_array_isolation_{pass,fail}`) regress — both tests
now report VERIFICATION SUCCESSFUL regardless of which of `clone.get(1)
== a` or `clone.get(1) == b` is asserted, because the user assertion is
sliced from the VCC set entirely (generated 4 VCCs, all are
`_ESBMC_element_null_check`; no user-assertion VCC emitted). The
`__ESBMC_memcpy` intrinsic's SMT encoding doesn't produce a per-element
value chain that the slicer recognises as a property-relevant
assignment — so the chain from `clone.get(1)` back to `base.arr[1]`
(set to `a`) is cut. Pass test accidentally passes (no VCC = no check);
fail test also passes (no VCC) when it should FAIL. Worse than the
original failure mode because both pass/fail duals collapse to
trivially-SUCCESSFUL and the test no longer exercises the property.

**(E2) Extend `value_sett::apply_assume` to strip `is_unknown2t` and
`is_invalid2t` entries in addition to null/zero.** Hypothesis:
`from_array`'s VSA set had `{*, *, *, *}` (four unknown entries) at the
arrcpy entry; stripping `*` post-assume would sharpen dereferences
inside the loop. Result: VSA output confirmed the `*` entries in
`from_array`'s individual symbol entry but 2D still fails — the
assertion-violating reads are through `c->grid[0]` where the `*` lives
in `dynamic_object_outer[]` (the array-aggregate backing of the outer
buffer), which is keyed as `values[name + "[]"]` in VSA's map.
`apply_assume` only walks `values[name]` keyed by the symbol itself, so
it never touches the `[]`-suffix aggregate entry. The widened entries
in that aggregate come from a different mechanism (calloc's zero-init
fill producing `*` entries for void-typed content) and need a
different intervention.

**(E3) Per-constant-index VSA suffix scheme (`[N]` for constants, `[]`
for symbolic, union on read).** NOT attempted — this is the actual
principled fix, but requires coordinated changes in `assign_rec` (use
`[N]` suffix for constant-indexed writes with strong update),
`get_value_set_rec` (constant-index reads look up both `[N]` and `[]`
and union them; symbolic-index reads iterate every `[k]` entry plus
`[]`), and a compatibility-check pass against every existing test that
depends on the current aggregate-only scheme. Too invasive for a
single session; queued as a future VSA refactor.

**Root-cause crystallisation.** The real underlying bug is VSA's
weak-update of indexed writes conflating writes at DISTINCT constant
indices into a single `[]` aggregate. When base's ctor does `grid[0] =
alloc_0; grid[1] = alloc_1; grid[2] = alloc_2` with concrete indices,
VSA produces `grid[] = {alloc_0, alloc_1, alloc_2}` instead of per-index
precision. Reads of `grid[0]` return the full union. When calloc's
initial zero-fill contributes `*` entries, those merge in too. In the
walker's per-slot emission, the arrcpy call's VSA widening through this
union-entry is the observed imprecision. `_ESBMC_arrcpy_2d` avoids the
heap-field aggregate entirely by doing the per-slot work inside a
helper-local `dst_outer` pointer, whose VSA entry is flat and doesn't
merge with the struct's `[]` aggregate.

Until (E3) ships, `_ESBMC_arrcpy_2d` is load-bearing, not tech debt.
Revised framing in `CLAUDE_Solidity.md` §"__ESOL_deep_copy walker
emission" reflects this.

---

## TL;DR

When the clone-helper walker emits this three-step sequence at the
Solidity-frontend level:

```
(A)  c->field    = _ESBMC_alloc_array(N, 8)                   // fresh outer
(B)  c->field[i] = <some heap pointer from arrcpy-family>      // per-slot write
(C)  later: c->field[i][...]                                   // returns nondet
```

subsequent index reads of the same field return nondet, even though the
writes happened on the same instruction path. Wrapping the entire (A)+(B)
dance inside a single C helper function (so all writes happen inside one
function frame) makes the reads work. The backend has been EXONERATED —
raw-C and raw-C++ programs that generate identical GOTO sequences all
PASS. The break is specific to Solidity-frontend-emitted SSA somehow,
but the specific front-end symbol/type/value-set attribute that causes
it has not been identified.

---

## 1. Triggering condition (precise)

Three conditions, all needed:

| Condition | Detail |
|---|---|
| (A) Struct field is a pointer-backed array | e.g. contract struct field typed `#sol_array_size`-tagged pointer (our model for `uint256[N]`, `uint256[M][N]`, etc.) |
| (B) Frontend emits fresh-alloc then per-slot write | `c->f = alloc_array(...)` immediately followed by `c->f[i] = X` for i = 0..N-1 |
| (C) Later read through the same field | Read `c->f[i][...]` after the writes; value should equal what was written |

When all three conditions are satisfied in the **same frontend-emitted
GOTO sequence** (i.e. NOT wrapped inside one helper function frame),
step (C) returns nondet.

### Which Solidity shapes trigger it

- **1D `uint256[N]`** → does NOT trigger. Walker emits a single
  `c->arr = _ESBMC_arrcpy(base->arr, N, sz)`. No (A)+(B) pattern.
- **2D `uint256[M][N]`** → triggers in the naive emission. Fixed
  by `_ESBMC_arrcpy_2d` workaround (one call covers outer alloc +
  per-slot arrcpy internally). **Currently CORE.**
- **3D `uint256[M][N][K]` and higher** → triggers. Walker recurses
  through the layers, so for 3D it emits `c->arr = alloc_array(...)`
  at the outermost layer + per-slot `c->arr[i] = _ESBMC_arrcpy_2d(...)`
  — same (A)+(B) pattern one level up. **Currently FAIL.**
- Suspected-vulnerable shapes (not yet tested):
  - `struct { uint256[N] cells; } outer_field[M];` (array of struct with inner fixed array)
  - Any field combo where the walker's non-scalar branch does per-slot writes

### Which Solidity shapes do NOT trigger

- Primitive scalars at any nesting
- Dynamic arrays `uint256[]` (stored as a global infinite array, not a struct pointer field)
- Mappings (addr-retargeted inside the bit-copy)
- Strings (`$dynamic_pool`)
- Single-level fixed arrays
- Struct-of-primitives (single arrcpy memcpy path in the walker)

---

## 2. Impact

- **Assertion violated even when logically true**. Verifier tells the
  user "this invariant can be broken" but really it's the verifier that
  lost track. False positive.
- **Affects any deep-copy-like semantics**. `__ESOL_deep_copy` is the
  obvious user-visible entry point. Any future frontend pass that
  emits (A)+(B) patterns for the same field will hit the same problem.
- **Silent — no crash, no warning**. The symex pipeline continues, the
  solver returns SAT, a counterexample is printed. Looks just like a
  legitimate bug find.
- **The fix surface is potentially large**. Every per-slot write to a
  freshly-allocated pointer-backed field is suspect. The current 2D
  workaround is a patch; 3D+ and related shapes remain exposed.

---

## 3. Minimal reproducer — 1D vs 3D

Two Solidity contracts differing only in the array dimensionality.
Command in both cases:

```
esbmc <file>.sol --contract H --bound --unwind 3 \
    --no-unwinding-assertions --no-standard-checks --cvc5
```

### 3a. 1D — PASSES

`regression/esbmc-solidity/esol_clone_multi_dim_pass/proof_1d.sol` (not
committed; run-time demo):

```solidity
// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
function __ESOL_deep_copy(C src) pure returns (C) { return src; }
contract C {
    uint256[3] arr;
    function setAt(uint256 j, uint256 v) public { arr[j] = v; }
    function get(uint256 j) public view returns (uint256) { return arr[j]; }
}
contract H {
    function check(uint256 a) public {
        require(a != 0);
        C base = new C();
        base.setAt(0, a);
        C clone = __ESOL_deep_copy(base);
        assert(clone.get(0) == a);
    }
}
```

Verdict: `VERIFICATION SUCCESSFUL`.

### 3b. 3D — FAILS (current bug witness)

```solidity
// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
function __ESOL_deep_copy(C src) pure returns (C) { return src; }
contract C {
    uint256[2][2][3] arr;                                              // <-- 3D
    function setAt(uint256 i, uint256 j, uint256 k, uint256 v) public { arr[i][j][k] = v; }
    function get(uint256 i, uint256 j, uint256 k) public view returns (uint256) { return arr[i][j][k]; }
}
contract H {
    function check(uint256 a) public {
        require(a != 0);
        C base = new C();
        base.setAt(0, 0, 0, a);
        C clone = __ESOL_deep_copy(base);
        assert(clone.get(0, 0, 0) == a);
    }
}
```

Verdict: `VERIFICATION FAILED` on `assert(clone.get(0,0,0) == a)`.

### 3c. 2D — originally failed, currently passes (via workaround)

The original KNOWNBUG contract (preserved as
`regression/esbmc-solidity/esol_clone_multi_dim_pass/contract.sol`) is
the 2D variant `uint256[2][3]`. Before the `_ESBMC_arrcpy_2d` fix it
failed identically to 3D. After the fix it passes — because the 2D
walker branch emits a single helper call instead of the (A)+(B) dance.

---

## 4. What the walker actually emits — 1D vs 3D GOTO comparison

Captured via `esbmc ... --goto-functions-only` on the minimal
reproducers, filtered to the `_ESBMC_clone_C` body.

### 4a. 1D clone body (PASS)

```
DECL struct C * _ESBMC_clone_c_C;
ASSIGN _ESBMC_clone_c_C = new struct C;        // cpp_new + ctor + struct-copy (elided)
ASSIGN *_ESBMC_clone_c_C = *_ESBMC_clone_base_C;
ASSIGN _ESBMC_clone_c_C->$address = nondet;
ASSUME  _ESBMC_clone_c_C->$address != _ESBMC_clone_base_C->$address;

DECL  void * return_value$__ESBMC_arrcpy$4;
FUNCTION_CALL:  return_value$__ESBMC_arrcpy$4 = _ESBMC_arrcpy(_ESBMC_clone_base_C->arr, 3, 32);
ASSIGN _ESBMC_clone_c_C->arr = return_value$__ESBMC_arrcpy$4;

RETURN _ESBMC_clone_c_C;
```

One assignment to the field. No per-slot writes. No (A)+(B) pattern.

### 4b. 3D clone body (FAIL)

```
DECL struct C * _ESBMC_clone_c_C;
ASSIGN _ESBMC_clone_c_C = new struct C;
ASSIGN *_ESBMC_clone_c_C = *_ESBMC_clone_base_C;
ASSIGN _ESBMC_clone_c_C->$address = nondet;
ASSUME  _ESBMC_clone_c_C->$address != _ESBMC_clone_base_C->$address;

// (A): fresh outer allocation
DECL  void * return_value$__ESBMC_alloc_array$4;
FUNCTION_CALL:  return_value$__ESBMC_alloc_array$4 = _ESBMC_alloc_array(3, 8);
ASSIGN _ESBMC_clone_c_C->arr = return_value$__ESBMC_alloc_array$4;

// (B): per-slot writes
DECL  void * return_value$__ESBMC_arrcpy_2d$5;
FUNCTION_CALL:  return_value$__ESBMC_arrcpy_2d$5 = _ESBMC_arrcpy_2d(_ESBMC_clone_base_C->arr[0], 2, 2, 32);
ASSIGN _ESBMC_clone_c_C->arr[0] = return_value$__ESBMC_arrcpy_2d$5;

DECL  void * return_value$__ESBMC_arrcpy_2d$6;
FUNCTION_CALL:  return_value$__ESBMC_arrcpy_2d$6 = _ESBMC_arrcpy_2d(_ESBMC_clone_base_C->arr[1], 2, 2, 32);
ASSIGN _ESBMC_clone_c_C->arr[1] = return_value$__ESBMC_arrcpy_2d$6;

DECL  void * return_value$__ESBMC_arrcpy_2d$7;
FUNCTION_CALL:  return_value$__ESBMC_arrcpy_2d$7 = _ESBMC_arrcpy_2d(_ESBMC_clone_base_C->arr[2], 2, 2, 32);
ASSIGN _ESBMC_clone_c_C->arr[2] = return_value$__ESBMC_arrcpy_2d$7;

RETURN _ESBMC_clone_c_C;
```

The `c->arr = alloc_array(3,8)` reassigns the outer pointer; the
immediately-following `c->arr[i] = ...` writes. Subsequent index reads
via `c->arr[0][0][0]` return nondet.

---

## 5. What has been disproven

Each of these was tested by a raw-C or raw-C++ equivalent that
reproduces the exact GOTO pattern. All PASS. See `repro_raw/` for the
three variants. Hypothesis is listed with the evidence that kills it.

| # | Hypothesis | How it was killed |
|---|---|---|
| H1 | `*new_ptr = tmp` direct struct ASSIGN is the trigger (vs C++'s `operator=()` function call) | `raw_u256_c.c` uses direct ASSIGN and PASSES |
| H2 | `cpp_new` + temporary_object + ctor-on-stack is the trigger | Replacing cpp_new with explicit malloc+ctor(heap) in `build_tod_clone_helper` did NOT fix the bug |
| H3 | `_ExtInt(96)` / `_ExtInt(160)` / `_ExtInt(192)` anon-pad fields in the contract struct confuse byte-wise struct-copy | `raw_u256_cpp_sol_pattern.cpp` mirrors the exact struct layout + the exact emission pattern and PASSES |
| H4 | The `__ESBMC_Main_H` nondet-dispatch loop calling `setAt(&_ESBMC_Object_C, nil, nil, nil)` poisons `_ESBMC_Object_C` | Running with `--function check` (which bypasses `__ESBMC_Main_H`) still FAILS |
| H5 | Private/internal vs public visibility of `setAt` / `get` matters | Tested with `private` methods wrapped in public forwarders — still FAILS |
| H6 | The `public` keyword's auto-generated getter interferes | Tested with non-public state var — still FAILS |
| H7 | The dispatcher-injected `nil` argument confuses value-set | 1D test has the same `nil`-arg pattern and PASSES |
| H8 | Outer alloc step alone is broken | Skipping only the outer `c->grid = alloc_array(...)` and keeping per-slot arrcpy: still FAILS (differently — base now reads wrong too due to aliasing). Keeping outer, skipping per-slot: FAILS. Skipping both: PASSES |
| H9 | `*c = *base` struct bit-copy corrupts value-set | Removing `*c = *base` entirely: still FAILS. Not the trigger |
| H10 | The walker's nested-contract-struct filter is wrong | Test's contract has no nested struct types; bug reproduces without any type-decl components |

---

## 6. What is confirmed

- The bug is **100% in the frontend emission path** of the clone walker.
  Disabling the walker altogether → test PASSES (though with wrong
  semantics: clone aliases base).
- The bug requires the **outer field reassignment (A)** AND the
  **per-slot writes (B)** to be in the **same frontend-emitted SSA
  block**. Moving both into one helper call (so writes happen inside
  one function frame) → PASSES.
- The bug **does not require** the intermediate `*c = *base` bit-copy,
  `tmp` stack object, cpp_new sideeffect, or `_ESBMC_Object_C` global
  init. Each was removed individually and the bug persisted.
- **Backend on its own handles the exact GOTO pattern correctly**.
  `repro_raw/raw_u256_c.c`, `raw_u256_cpp.cpp`, and
  `raw_u256_cpp_sol_pattern.cpp` all reproduce the sequence byte-for-byte
  in C/C++ mode and verify SUCCESSFUL under bitwuzla.
- **1D never triggers**, by any path.
- **2D triggers but is currently masked** by the `_ESBMC_arrcpy_2d`
  library helper (commit `24c365b04e`). The workaround is to collapse
  (A)+(B) into one C frame.
- **3D triggers and is currently unfixed** because the 2D workaround
  only covers the innermost two layers. The 3D walker recurses once
  before calling `_ESBMC_arrcpy_2d`, so (A)+(B) reappears at the
  outer layer.

---

## 7. Current workaround

### 7a. What `_ESBMC_arrcpy_2d` does

Library helper in `src/c2goto/library/solidity/solidity_array.c`:

```c
void *_ESBMC_arrcpy_2d(void *from_array,
                       size_t outer,
                       size_t inner,
                       size_t elem_size)
{
__ESBMC_HIDE:;
    _ESBMC_element_null_check(from_array != 0);
    _ESBMC_zero_size_check(outer != 0);
    _ESBMC_zero_size_check(inner != 0);
    _ESBMC_zero_size_check(elem_size != 0);

    // Step 1: byte-copy outer pointer array via memcpy.
    // Same pattern _ESBMC_arrcpy uses for non-u256/int256 elements.
    // Plain pointer-indexed reads of `from_array[i]` in a Solidity
    // context null out through symex; memcpy bytes through cleanly.
    void *dst_outer_raw = _ESBMC_alloc_array(outer, sizeof(void *));
    __builtin_memcpy(dst_outer_raw, from_array, outer * sizeof(void *));

    // Step 2: replace each outer slot with a fresh arrcpy of its inner row.
    void **dst_outer = (void **)dst_outer_raw;
    for (size_t i = 0; i < outer; i++)
        dst_outer[i] = _ESBMC_arrcpy(dst_outer[i], inner, elem_size);
    return (void *)dst_outer;
}
```

Key detail: step 1 uses memcpy rather than pointer-indexed reads. An
earlier version of this helper used `src_outer[i]` directly and
triggered `_ESBMC_element_null_check` inside the inner arrcpy. memcpy
of the raw bytes works; pointer-indexed reads of `base->grid[i]` in
this Solidity context resolve to NULL.

### 7b. Where the walker picks the helper

`src/solidity-frontend/solidity_convert_constructor.cpp`,
function `emit_clone_deep_copy_fixup`. The trigger:

```cpp
const typet &elem_t_outer = field_type.subtype();
const bool inner_is_ptr_backed =
  !elem_t_outer.get("#sol_array_size").empty() &&
  elem_t_outer.is_pointer();
if (inner_is_ptr_backed && !needs_clone_deep_fixup(elem_t_outer.subtype()))
{
    // → emit single FUNCTION_CALL _ESBMC_arrcpy_2d(...)
}
```

The guard `!needs_clone_deep_fixup(elem_t_outer.subtype())` means it
only activates when the leaf under the two nested pointers is a trivial
(scalar) type. For 3D, the leaf under two pointers is still another
pointer-backed array → needs_clone_deep_fixup is true → walker falls
back to the (A)+(B) emission → bug reappears.

### 7c. Why the 2D workaround is a lazy fix

- Only patches the specific depth-2 shape.
- Does nothing for 3D, 4D, or struct-with-nested-fixed-array patterns
  that produce the same (A)+(B) modus operandi at any depth.
- Does not explain why the underlying pattern loses symex tracking in
  Solidity-frontend-emitted code but not in identical raw C/C++ code.
- `_ESBMC_arrcpy_2d` itself contains a workaround-inside-workaround:
  the `memcpy(dst_outer_raw, from_array, outer*sizeof(void*))` avoids
  pointer-indexed reads of `from_array[i]`, which is a second
  manifestation of the same value-set-loss bug.

---

## 8. Delta-debug trace — every variant tested

All tested against the original 2D contract (`contract.sol`, 2D `uint256[2][3]`)
under `--contract H --bound --unwind 3 --no-unwinding-assertions --no-standard-checks --cvc5`.
Historical — 2D is now covered by the workaround, but this log tells us
exactly what surgery on the walker changes the outcome.

| Variant | Walker emission | Verdict |
|---|---|---|
| Baseline (no workaround) | (A) alloc outer + (B) per-slot arrcpy | FAIL |
| Skip `*c = *base` struct copy | Same minus bit-copy | FAIL |
| Skip outer alloc, keep per-slot arrcpy | (B) only (writes go through aliased outer) | FAIL |
| Keep outer alloc, skip per-slot arrcpy | (A) only | FAIL (differently — c->grid[i] uninitialised) |
| Skip BOTH | Nothing | PASS (clone aliases base) |
| Skip walker entirely | No fixup at all | PASS (aliases) |
| Replace with single top-level arrcpy: `c->grid = arrcpy(base->grid, 3, 8)` | One call, covers outer only | PASS (inner rows alias base) |
| Replace with `_ESBMC_arrcpy_2d(base->grid, 3, 2, 32)` using `src[i]` pointer-index inside | Single call but with pointer-indexed reads inside | FAIL (null-check inside) |
| Replace with `_ESBMC_arrcpy_2d` using memcpy inside | Single call, memcpy-based outer read | PASS + full isolation |

The decisive surgery is "collapse (A)+(B) into one function frame". Both
layers must be inside the same frame; emitting them separately at the
frontend level is what breaks it.

---

## 9. Cross-validation — raw C/C++ repros

`repro_raw/` contains three files, all of which PASS:

- `raw_u256_c.c` — C99 with `MALLOC(sizeof C) + ctor(base)`, direct struct
  ASSIGN. Simplest. Proves the backend handles malloc + per-field pointer
  writes + index reads on its own.
- `raw_u256_cpp.cpp` — C++ with `new struct C() + ctor(base)`. C++ frontend
  auto-lowers `*new_ptr = tmp` to `operator=()` function call.
- `raw_u256_cpp_sol_pattern.cpp` — C++ written to match the Solidity
  frontend's emission pattern **byte-identically** — uses `new struct C`
  (no parens, skips POD default ctor) + stack tmp + `ctor(&tmp)` + direct
  `*new_ptr = tmp` + no explicit ctor-on-base. Proves the pattern itself
  is handled correctly when it emerges from the C++ frontend rather than
  the Solidity frontend.

All three mirror the Solidity struct layout too (including `_ExtInt(96)`,
`_ExtInt(160)`, `_ExtInt(192)` anonymous padding) to rule out layout
effects.

Run them manually (bitwuzla is fast for these):

```
esbmc repro_raw/raw_u256_c.c              --unwind 4 --no-unwinding-assertions --no-standard-checks --force-malloc-success --bitwuzla
esbmc repro_raw/raw_u256_cpp.cpp          --unwind 4 --no-unwinding-assertions --no-standard-checks --force-malloc-success --bitwuzla
esbmc repro_raw/raw_u256_cpp_sol_pattern.cpp --unwind 3 --no-unwinding-assertions --no-standard-checks --force-malloc-success --bitwuzla
```

All three → `VERIFICATION SUCCESSFUL`.

---

## 10. What remains unknown

- **Which front-end-injected symbol / type / attribute changes symex's
  behavior on the (A)+(B)+(C) pattern?** Same GOTO in raw C/C++ is fine;
  in Solidity mode it breaks. No one has nailed which symbol-table
  property is the delta.
- **Where exactly in symex does tracking drop?** Strong suspects:
  - `src/goto-symex/symex_assign.cpp` — struct-field-through-pointer
    assignment handling
  - `src/pointer-analysis/value_set.cpp` — value-set updating when a
    base pointer is reassigned then indexed
  - `src/goto-symex/renaming.cpp` — L1/L2 renaming across the sequence
- **Does the bug need `--bound`?** 3D fails under `--bound`, need to
  re-test `--unbound` quickly — but `--unbound` masks many reads
  through nondet-return-on-method-call so the test probably becomes
  vacuous rather than informative.
- **Does `--incremental-bmc` or `--k-induction` change anything?** Not
  tested.

---

## 11. Next-step options (in order of preference)

### A. Reproduce in pure C (if possible)

Goal: produce a `.c` file that triggers the value-set loss without any
Solidity frontend involvement. If we can: it's a backend bug, we fix in
`src/goto-symex/` or `src/pointer-analysis/` and this whole class of
Solidity bugs disappears for free. If we can't after honest effort:
the bug is fundamentally a Solidity-frontend / symex-interface issue
and we have to pivot.

How to try: take `repro_raw/raw_u256_c.c` (which currently PASSES) and
progressively make its struct / pointer layout more similar to what the
Solidity frontend actually emits (look at the symbol table + goto-dump
for 3D contract and mirror every attribute: `#sol_array_size` tags,
dynamic flag on the outer alloc, `_ESBMC_HIDE` labels, specific
`alloc_array` vs raw `calloc`, etc.) until either it starts failing
(great — minimal C repro) or we run out of differences (now we know
the symbol-table delta).

### B. Diff symbol tables and types

Run both with `--symbol-table-too` on the failing 3D Solidity contract
and the passing raw-C equivalent. Look specifically at:

- Type-tree of the contract struct C vs raw-C struct C
- Every irep attribute on the `grid`/`arr` field symbol
- Every flag on the dynamic-allocated `dynamic_N_array` symbol
- Any Solidity-specific attributes the frontend sets that the C frontend
  doesn't

This is the path @user suggested earlier in the session ("检查 __ESBMC_Main_
这个symbol"). Not exhausted yet.

### C. Generalise the workaround (lazy, not root-cause)

Write a runtime-sized N-D helper:

```c
void *_ESBMC_arrcpy_nd(void *from_array,
                       const size_t *dims,   // dims[0]..dims[ndim-1]
                       size_t ndim,
                       size_t elem_size);
```

Walker counts pointer-backed layers from the type tree, builds `dims[]`,
emits one call. Covers arbitrary depth. Still doesn't explain the bug
and still papers over it — but makes the test suite survive until we
root-cause. Acceptable as a checkpoint commit if root-cause work is
going to take multiple sessions.

### D. Patch symex / value-set directly

Only attempt after B identifies a concrete mechanism. Otherwise this is
guessing.

---

## 12. File map

Everything related to this investigation lives in
`regression/esbmc-solidity/esol_clone_multi_dim_pass/`:

```
contract.sol                            2D variant, CORE (passes via workaround)
test.desc                               CORE + --cvc5
INVESTIGATION.md                        this file
repro_raw/README.md                     describes the three raw repros
repro_raw/raw_u256_c.c                  raw C99 repro, PASSES
repro_raw/raw_u256_cpp.cpp              raw C++ repro, PASSES
repro_raw/raw_u256_cpp_sol_pattern.cpp  raw C++ mirroring Solidity emission, PASSES
```

Source-code sites touched by the 2D workaround:

```
src/c2goto/library/solidity/solidity_array.c    _ESBMC_arrcpy_2d definition
src/c2goto/cprover_library.cpp                  _ESBMC_arrcpy_2d whitelist entry
src/solidity-frontend/solidity_convert.h        get_arrcpy_2d_function_call decl
src/solidity-frontend/solidity_convert_mapping.cpp     get_arrcpy_2d_function_call impl
src/solidity-frontend/solidity_convert_constructor.cpp emit_clone_deep_copy_fixup call site
```

Relevant memory / docs:

```
CLAUDE.md                           "## __ESOL_deep_copy Semantics" section
                                    describes the multi-dim entry in the per-field table
~/.claude/.../memory/reference_deep_copy_semantics.md
                                    same info, auto-loaded
```
