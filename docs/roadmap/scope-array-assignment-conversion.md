# Scope — the array-typed assignment conversion

> **Status: Phase 1 shipped (#6700), Phase 2 gates discharged 2026-08-05 (§12).
> The scope's own work is done; the flip it contributes to is not.**
> `docs/roadmap/scope-coupled-arith-assign-conversion.md` §13.1 discharges every
> gate that scope owns and closes with the flip blocked on **two mechanisms it
> explicitly does not own**. This is the owner document for one of them. The
> other — §9.4's "second mechanism" (`chained-comparison2_fail`, `lambda15`,
> `precedence2`, `sum_tuple`) — is owned by
> `scope-relational-float-reconciliation.md`, reopened at its §17 with one
> witness clearing and nothing shipped.
>
> **Verification status of this document.** The tree reads in §3 were performed
> against `master` on 2026-08-03. The GOTO diff and abort message in §2 are
> **inherited from that scope's §14**, a recorded measurement, and were *not*
> re-verified here: the dev machine was contended (load ~14) and a full symex
> run could not complete. Re-run the §2 commands before acting — §7 G0 exists
> for exactly that.

## 1. What this unblocks, and what it does not

- **Unblocks** one of the two remaining blockers on Phase 3 of the coupled-arith
  scope — the `python_adjust` flip (`--python-irep2-adjust-only` becoming the
  default). Clearing it is necessary but **not sufficient**: the §9.4 second
  mechanism is independent and unowned.
- **Does not** move any §V.1 acceptance bar. Bars #1/#2/#4 are V.2/W3 and the
  symbol-write surface, tracked in `irep2-migration.md` §V.7 and
  `frontends-to-irep2.md`.
- Blast radius is the Python path only while the flag stays default-off.

## 2. The defect (inherited measurement — re-verify per G0)

Witnesses: `regression/python/github_5571_fail` and
`regression/python/github_5571_tuple_str_annotation`. Under
`--python-irep2-adjust-only` both abort:

```
ERROR: Typecast for unexpected type
typecast
* from : constant_array   (char[1], i.e. the "" literal + NUL)
* type : array            (char[16])
```

Source shape (verified — `github_5571_tuple_str_annotation/main.py`):

```python
def f(pairs):
    s = ""
    for u, v in pairs:
        s = v
    return s
```

`s` carries the #5571 fixed-width tuple-string representation, `char[16]`; the
`""` literal is `char[1]`. The whole difference in `f` is one line:

```
legacy:   DECL signed char [0] s;  ASSIGN s = (signed char [16])(&{ 0 }[0]);
hop-off:  DECL signed char [0] s;  ASSIGN s = { 0 };
```

Legacy does **two** things at the assignment seam — decays the `char[1]`
literal to `&{0}[0]`, then casts that pointer to `char[16]`. The hop-off does
neither and assigns the bare `constant_array`.

**The offending cast is not in the hop-off GOTO at all.** It is synthesised
later, during symex, when the `char[1]` value meets its differently-sized
destination — and `convert_typecast` has no array arm, hence the abort. This is
corroborated here: under `--python-irep2-adjust-only --goto-functions-only` the
test completes normally, so nothing in GOTO construction fails.

So the defect class is the one the coupled scope's §2 describes — an
unconverted assignment reaching the solver — with an **array** type instead of
a scalar one.

## 3. Why the existing arms decline it (verified against the tree)

| arm | guard | why it declines |
|---|---|---|
| array→pointer decay (`python_adjust.cpp:473-474`) | `is_typecast2t(expr) && is_pointer_type(expr->type) && is_array_type(from->type)` | requires a cast **to a pointer**; here the target is an **array** type |
| Phase 2 assignment arm (coupled scope §12.1) | numeric-to-numeric, plus pointer-source-into-Boolean | an array source into an array target matches neither clause |

**The node kind is not the obstacle.** The coupled scope's §11 measured that
every Python-source assignment, plain or augmented, arrives as `code_assign2t`;
its §14 confirms the assignment here is one. What declines it is the guard.

## 4. What an array arm has to reproduce

Legacy's shape is a cast of a **pointer** to an **array** type, assigned to a
variable whose declared type is `char[0]`:

```
(signed char [16])(&{ 0 }[0])
```

All three widths differ — 1 (source literal), 16 (cast target), 0 (declared).
The existing decay arm cannot emit it: it produces an `address_of`, never an
array-typed cast.

## 5. Options

| # | Option | Where | Verdict |
|---|---|---|---|
| A | Array-aware assignment arm in `python_adjust` — reproduce legacy's decay-then-cast | Python frontend | **Likely**, but see §6 Phase 0 — it may be treating a symptom |
| B | Give `convert_typecast` an array arm | SMT layer | Fixes the abort for *every* frontend, but the blast radius is the solver backend and it legitimises a same-value/different-width array cast |
| C | Emit a correctly-sized literal at the converter — `s = ""` where `s` is `char[N]` builds a `char[N]` zero-filled constant, not `char[1]` | Python converter | Removes the mismatch instead of converting it; may be the actual bug |

**Do not pick before Phase 0.** A, B and C treat three different layers, and
the coupled scope's own history is a warning: its §13 concluded the pair was "a
third mechanism this scope cannot reach", and §14 then showed that was half
wrong. The same discipline applies here.

## 6. Phased decomposition

### Phase 0 — locate the defect (no code change)
Answer one question: **is the `char[1]` literal wrong, or is the missing
conversion wrong?** Compare what the converter builds for `s = ""` when `s` is
`char[16]` against what it builds for a same-width assignment, on both the
legacy and hop-off paths. Deliverable: a recorded answer naming A, B or C, with
the GOTO evidence. *Accept:* the §2 measurement re-verified (G0) and the layer
identified.

### Phase 1 — implement the chosen option
Behind the existing `--python-irep2-adjust-only` flag, which stays default-off.

### Phase 2 — contribute to the flip gate
Re-run the coupled scope's G1/G2/G3 plus this scope's G4; hand the result back
to that document's Phase 3.

## 7. Gates

| # | Gate |
|---|---|
| **G0** | §2's abort and GOTO diff reproduce on the current tree. This document's §2 is inherited, not re-verified |
| G1 | `github_5571_fail` and `github_5571_tuple_str_annotation` produce **legacy-identical verdicts** under the hop-off |
| G2 | The coupled scope's anti-masking gate still holds — `neural-net_fail` (`--fixedbv`) reports FAILED. A conversion added at an assignment seam is exactly the shape that masked it once |
| G3 | Default path unaffected: the flag is default-off, so this is by construction; assert it with a default-path slice anyway |
| G4 | **The array/pointer mismatch at symex rename does not return.** The decay arms were added to fix it; an array-typed cast is the neighbouring shape. Cite the tests that pinned it and run them |
| G5 | Dual-solver agreement (Bitwuzla + Z3) |

## 8. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Getting the emitted shape wrong reintroduces the symex-rename array/pointer mismatch | G4; the coupled scope's §14 flags this explicitly as the reason it was not attempted blind |
| R2 | Option B changes the solver backend for all five frontends | Phase 0 must justify it before it is taken; prefer A or C |
| R3 | Fixing the symptom (A) leaves a wrongly-sized literal in the IR that bites elsewhere | Phase 0 asks the question directly; C exists for this answer |
| R4 | §2 is inherited and the tree has moved | G0 is the first gate for this reason |

## 9. Non-goals

- The §9.4 second mechanism (`chained-comparison2_fail`, `lambda15`,
  `precedence2`, `sum_tuple`). Different mechanism, still unowned, still needs
  its own scope document.
- The coupled scope's own gates — discharged, see its §13.1.
- Anything in `frontends-to-irep2.md` Phase 1; that is the shared
  `goto_convert` dispatcher, not the Python adjuster.

## 10. One-line summary

An assignment of a `char[1]` literal to a `char[16]` destination reaches symex
unconverted, where a synthesised array-to-array typecast meets a
`convert_typecast` with no array arm — the coupled scope's defect class with an
array type, declined by every existing arm because they all guard on a pointer
target.

## 11. Phase 0 — the layer is the adjuster (2026-08-04)

§6 Phase 0 asks one question before any code: **is the `char[1]` literal wrong,
or is the missing conversion wrong?** §5 lists three candidate layers and
forbids choosing before this is answered.

### 11.1 Measurement

`github_5571_tuple_str_annotation`, `--goto-functions-only`, both paths:

```
legacy:   DECL signed char [0] s;   ASSIGN s = (signed char [16])(&{ 0 }[0]);
hop-off:  DECL signed char [0] s;   ASSIGN s = { 0 };
```

§2's inherited diff is reproduced exactly, and G0 is discharged.

The `char[0]` declaration looks like the defect at first glance — three widths
are in play (declared 0, cast target 16, literal 1). **It is not.** The control
settles it:

| test | declares `s` as | default path |
|---|---|---|
| `github_5571` | `signed char [0]` | **passes** |
| `github_5571_fail` | `signed char [0]` | passes (as a `_fail`) |
| `github_5571_tuple_str_annotation` | `signed char [0]` | passes; **aborts only under the hop-off** |

Every variant declares `char[0]`, including the ones that verify correctly on
the legacy path. So the zero-width declaration is the intended representation —
the assignment's cast is what carries the real width — and it is not what
distinguishes the failing configuration.

### 11.2 Answer

**The missing conversion is the defect; the literal and the declaration are
both fine.** The only difference between a working run and an aborting one is
that legacy emits the decay-then-cast at the assignment seam and the hop-off
emits neither.

That selects **§5 Option A — an array-aware assignment arm in `python_adjust`**,
and rules out Option C (a correctly-sized literal at the converter), which would
change a representation the working path depends on. Option B (an array arm in
`convert_typecast`) remains available but is now clearly the wrong layer to
start at: it would make the solver tolerate a cast the frontend should not be
failing to emit.

### 11.3 What Phase 1 must reproduce

Unchanged from §4, now with the layer fixed:

```
(signed char [16])(&{ 0 }[0])
```

a cast of a **pointer** to an **array** type, assigned to a `char[0]`-declared
variable. §4's warning stands — the existing decay arm emits an `address_of`
and cannot produce this — as does R1: getting the shape wrong risks the
array/pointer mismatch at symex rename that the decay arms were added to fix.

## 12. Phase 2 — the gates, discharged (2026-08-05)

Phase 1 shipped as **#6700** (`c5efabb9c1`): an `is_constant_array2t`-guarded
arm in `python_adjust`'s general assignment conversion that decays the literal
to `&lit[0]` and casts that pointer to the target array type. That PR claimed
G1, G2 and G3 and explicitly did **not** claim G4 or G5. Both are discharged
here, and G1/G3 are re-measured on a wider slice.

### 12.1 G5 — dual-solver agreement

Both witnesses, `--python-irep2-adjust-only --incremental-bmc`:

| witness | Bitwuzla | Z3 |
|---|---|---|
| `github_5571_tuple_str_annotation` | SUCCESSFUL | SUCCESSFUL |
| `github_5571_fail` | FAILED | FAILED |

### 12.2 G4 — the array/pointer mismatch has not returned

G4 asks for the tests that pinned the symex-rename mismatch the decay arms were
added to fix. #6363's commit message names five; all five agree with legacy
under the hop-off, and none of them is the witness this scope fixed:

| test | legacy | hop-off |
|---|---|---|
| `string-concat6` | SUCCESSFUL | SUCCESSFUL |
| `string-concat11` | SUCCESSFUL | SUCCESSFUL |
| `github_3090_4_fail` | FAILED | FAILED |
| `string18` | SUCCESSFUL | SUCCESSFUL |
| `string22_fail` | FAILED | FAILED |

The 56 `python_irep2_adjust*` ctest cases — the suite that exercises the
hop-off directly, including `python_irep2_adjust_only_array_assign{,_fail}`
added by #6700 — pass 56/56.

### 12.3 G1 and G3, re-measured wider

- **G1**: 30-test hop-off-vs-legacy verdict parity over the array-carrying
  families (`bytes*`, `str_*`, `concat*`, the three `github_5571` variants),
  each replayed with its own `test.desc` flags. **30/30 agree**, positives and
  `_fail` negatives alike.
- **G2**: `neural-net_fail` (`--fixedbv`) reports FAILED on both paths — the
  anti-masking gate re-confirmed against the shipped arm, not just the
  experimental one.
- **G3**: 317-test default-path ctest slice over the same families, 317/317
  pass. This is assertion, not proof: `python_adjust` runs only under
  `python_language.cpp:299`'s flag, so the default path cannot reach the arm.

### 12.4 The shipped guard is narrower than the conversion permits

The arm requires `is_constant_array2t(a.source)`, while
`check_c_implicit_typecast` permits any array source into an array target of
the same element type (`c_typecast.cpp:258-268`). The gap is a **non-constant**
array source meeting a differently-typed array target.

It has no witness. The same function contains one — `s = v` in the loop body —
and both paths emit it bare, identically, because the types already agree:

```
legacy:   ASSIGN s=(signed char [16])(&{ 0 }[0]);   ASSIGN s=v;
hop-off:  ASSIGN s=(signed char [16])(&{ 0 }[0]);   ASSIGN s=v;
```

Widening the guard to the full permission rule is therefore a change with no
demonstrated benefit, which is the shape
`scope-relational-float-reconciliation.md` §16.2 declined to ship. Left as
recorded scope, not as work.

### 12.5 A trap this re-measurement walked into

§2's abort reproduces perfectly on a binary built before `c5efabb9c1` — the
defect is fixed in the tree while the artefact still exhibits it. Re-deriving
Phase 0's answer from that binary produced a patch that duplicated the shipped
arm. **Check the binary's provenance against the fix commit, not just its
mtime**, before treating a reproduction as evidence the defect is open.

### 12.6 What this scope hands back

`scope-coupled-arith-assign-conversion.md` Phase 3 owns the flip. Of its two
blockers this one is now **cleared**; the other is
`scope-relational-float-reconciliation.md` §17, where one of four witnesses
clears and the widening that achieved it is still too broad to ship.

## 13. A second declined shape — an array source into a *scalar* target (2026-08-06)

§12 discharged this scope's gates for the array→**array** assignment. The G4
whole-corpus census (`scope-relational-float-reconciliation.md` §18.4) has since
surfaced a second shape that every arm also declines, with a harder failure
mode: the hop-off **aborts in the solver** where legacy verifies.

### 13.1 The witnesses and a 7-line reduction

`class10` and `class12` both carry a mutable class attribute appended through one
instance and read through another. Reduced to:

```python
class C:
    shared: list = []

a = C()
b = C()
a.shared.append("x")
assert "x" in b.shared
```

Under `--unwind 19 --no-standard-checks --smt-during-symex --smt-symex-guard
--bitwuzla`: legacy **SUCCESSFUL**, hop-off aborts with

```
bitwuzla: error: ... terms with mismatching sort at indices 0 and 1
```

### 13.2 The node, isolated against a passing control

The reduction's GOTO diff shows three differences, and a **passing** control
(`xs = ["a", "b"]; assert "a" in xs`, SUCCESSFUL on both arms) carries two of
them. That control is what makes the attribution safe rather than plausible:

| # | shape | in the passing control? | verdict |
|---|---|---|---|
| 1 | `ASSERT (signed int)contains_tmp == 1` vs `ASSERT contains_tmp == 1` (a `_Bool` compared to `1`, unpromoted) | **yes** | benign here |
| 2 | `ASSIGN v = (unsigned long int)(&{ 120, 0 }[0]);` vs `ASSIGN v = { 120, 0 };` | **no** | **the cause** |
| 3 | `list_push(..., &elem, ...)` vs `list_push(..., &elem[0], ...)` | **yes** | benign here |

Shape 2 is the only difference the aborting program has and the passing one does
not. `v` is `unsigned long int`; the source is a `signed char [2]` string
literal. Legacy does the same two steps §14 of the coupled-arith scope recorded
for the array→array case — decay to `&lit[0]`, then cast — but the **target here
is a scalar**, so the result is a pointer-to-integer cast rather than an array
cast. The hop-off assigns the bare `constant_array` into an integer lvalue, and
the solver is handed an array term where a bitvector is expected.

Shape 1 is worth a separate note, stated with the ambiguity it still carries. A
`_Bool` compared against `1` reaches the solver unpromoted on the hop-off and
does **not** abort. Whether that is a declined promotion or a benign
representational difference **cannot be read off the dump**: this format prints
a Boolean constant as `1`, not `TRUE` — there is no `TRUE`/`FALSE` token
anywhere in either dump — so `contains_tmp == 1` is equally consistent with

- `bool == bool`, which is well-sorted, and which the PR #6702 equality arm
  *correctly* declines: its guard requires one side Boolean and the other a
  number; or
- `bool == signedbv`, which that arm should convert and demonstrably does not,
  since `c_implicit_typecast_arithmetic` raises both operands to `INT` through
  `implicit_typecast_arithmetic`'s minimum promotion and would emit the cast.

Legacy's `(signed int)` on both sides is what that same minimum promotion
produces from *either* starting point, so it does not discriminate either.
Settling this needs the operand type — instrumentation and a rebuild. Recorded
as an open question, not as a defect, and not what breaks `class10` either way.

### 13.3 Why every arm declines it

The general assignment arm's guard (`python_adjust.cpp:642-649`) requires both
sides numeric, and an array source is not; its `bool`-target/pointer-source
disjunct does not apply either. §12's array→array arm requires an **array
target**. So the shape falls through every arm to no conversion at all.

The fix belongs to this scope — same seam, same two legacy steps — but the cast
target is the scalar type rather than the array type, and it must not be bolted
onto the array→array arm's guard: that arm's `is_constant_array2t` source test
is right for both, while its `is_array_type(target)` test is exactly what
excludes this one.

### 13.4 Not yet done

No patch. The reduction above is the regression test the fix should carry, and
it needs a `_fail`-free positive form plus the dual-solver run this scope's
Gates require.
