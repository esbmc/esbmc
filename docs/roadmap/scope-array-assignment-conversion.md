# Scope — the array-typed assignment conversion

> **Status: forward plan, opened 2026-08-03. Not started.**
> `docs/roadmap/scope-coupled-arith-assign-conversion.md` §13.1 discharges every
> gate that scope owns and closes with the flip blocked on **two mechanisms it
> explicitly does not own**. This is the owner document for one of them. The
> other — §9.4's "second mechanism" (`chained-comparison2_fail`, `lambda15`,
> `precedence2`, `sum_tuple`) — still has none.
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
