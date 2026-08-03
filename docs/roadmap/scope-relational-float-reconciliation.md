# Scope — relational/equality reconciliation over mixed float and integer operands

> **Status: forward plan, opened 2026-08-03. Not started.**
> This is the owner document for the mechanism
> `docs/roadmap/scope-coupled-arith-assign-conversion.md` §9.4 named the
> "second mechanism" and §7 explicitly disowned:
> `chained-comparison2_fail`, `lambda15`, `precedence2`, `sum_tuple`. With
> `scope-array-assignment-conversion.md`, this closes the last of the two
> unowned blockers on the `python_adjust` flip.
>
> **Verification status.** §2 (test sources) and §3 (the guards, with line
> numbers) were read from `master` on 2026-08-03 and are verified. §4 is a
> **hypothesis** that explains all four witnesses but has *not* been confirmed
> by instrumentation — the dev machine was contended (load ~19) and no
> corpus run could complete. Phase 0 exists to confirm or refute it, and it
> must run before any code is written.

## 1. What this unblocks

Phase 3 of the coupled-arith scope — the `python_adjust` flip
(`--python-irep2-adjust-only` becoming the default). That scope discharged
every gate it owns (§13.1); what remains is this mechanism plus the
array-typed assignment. Neither moves a §V.1 acceptance bar.

## 2. The four witnesses, and what they have in common (verified)

All four are insensitive to every arm the coupled scope prototyped — identical
in the bare hop-off and in each configuration (§9.4) — so they are not
stragglers of that mechanism.

| test | shape | mixed-kind operands? |
|---|---|---|
| `chained-comparison2_fail` | `x = 2.5` then `0 <= x <= 2 < 3` | **yes** — `x` is floatbv, the bounds are integer literals |
| `lambda15` | `between = lambda x: 0 <= x <= 10`; `is_even = lambda x: x % 2 == 0` | chained relational, and an **equality** |
| `precedence2` | `x /= 3` monomorphises `x` to double; later `x \|= 7` | **yes** — bitwise on a floatbv-typed variable |
| `sum_tuple` | `sum((1, 2.5, 3))` — int/float folding over a tuple | **yes** |

The common thread is **an integer and a floating-point operand meeting at a
comparison, a chained comparison, or a bitwise/augmented assignment**, not a
single syntactic construct. That is why per-case triage kept re-homing them.

`precedence2`'s localisation from §9.4 is the sharpest evidence:

```
legacy:   ASSIGN x=(double)((signed long int)((signed int)x) | 7);
hop-off:  ASSIGN x=(signed long int)((signed int)x) | 7;
```

## 3. The guards, read from the tree (verified)

`python_adjust.cpp` has two reconciliation arms with **asymmetric** operand
admission:

| arm | line | admits |
|---|---|---|
| relational (`<`, `<=`, `>`, `>=`) | `:406-409` | `ops.size() == 2 && is_bv_type(op0) && is_bv_type(op1) && signedness differs` |
| binary arithmetic (`+ - * / % & ^ \|`) | `:449` | `ops.size() == 2 && convertible(op0) && convertible(op1)`, where `convertible` is **bv or floatbv or fixedbv** (`:444-447`) |

Two consequences, both verified by reading:

1. **The relational arm excludes floating point entirely.** `is_bv_type` is
   false for `floatbv`, so `0 <= x` with `x` floatbv is never reconciled —
   regardless of signedness, the guard's third clause is never reached.
2. **There is no `equality2t`/`notequal2t` arm at all.** Legacy routes `==` and
   `!=` through `adjust_expr_rel` alongside the four ordering relations
   (`clang_c_adjust_expr.cpp:109-115`); `python_adjust` handles only the four.
   The coupled scope records this gap in its §1 and §9.3 and judged it "not the
   defect class below" — correctly, for *that* class.

## 4. Hypothesis (NOT yet confirmed)

> The second mechanism is the relational arm's `is_bv_type` admission plus the
> missing equality arm: a comparison or bitwise assignment with one floatbv and
> one integer operand reaches the solver unreconciled.

It explains all four witnesses and nothing else in the corpus needs to explain
them. It is consistent with §12.2's refutation of the node-kind theory for
`precedence2` — the node kind was never the obstacle, the operand *kinds* are.

**It is a hypothesis.** Confirming it is Phase 0's entire job, and Phase 0 must
not be skipped: the coupled scope's §13→§14 history is a confidently-argued
"cannot reach" conclusion that turned out half wrong.

## 5. The wall this must not walk into

**Widening the relational arm naively has already been tried and rejected.**
`python_adjust.cpp:396-402` records it: running `gen_typecast_arithmetic` on
*every* relational node "diverges corpus-wide from clang's promotions over the
OM bodies" (`scope-v1k-adjuster.md`, "gap-2"). The signedness-mismatch gate
exists precisely to exclude that traffic, leaving only nodes that are otherwise
a hard abort.

So the change this scope needs is **not** "drop the guard". It is a narrower
widening — admit a *float-versus-integer* pair while continuing to decline the
same-kind width promotions (char vs int) that gap-2 showed must stay untouched.
Whether that narrower rule is expressible without re-opening gap-2 is the
second question Phase 0 must answer.

## 6. Phased decomposition

### Phase 0 — confirm or refute the hypothesis (no code change)
Instrument the relational and equality dispatch in `python_adjust::adjust_expr`
to log operand type kinds for every comparison node, run the four witnesses
plus a corpus slice, and answer:

1. Do all four witnesses reach a comparison or bitwise node with one floatbv
   and one bv operand? (Confirms or refutes §4.)
2. How much corpus traffic would a float-vs-int admission rule touch, and how
   much of it is OM-body traffic of the kind gap-2 forbids?
3. Does the equality gap contribute independently of the floatbv gap, or are
   they the same nodes?

*Accept:* a recorded answer with counts. A refutation is a success — it
re-homes these four again, with evidence this time.

### Phase 1 — the narrowed admission rule
Only if Phase 0 confirms. Widen the relational guard to admit a float/integer
pair, add the equality/notequal arm if Phase 0 shows it contributes, and keep
declining same-kind width promotions.

### Phase 2 — hand back to the flip
Re-run the coupled scope's G1-G3 and this scope's G4-G5; report to that
document's Phase 3.

## 7. Gates

| # | Gate |
|---|---|
| **G0** | Phase 0's census exists and §4 is confirmed or refuted, with counts |
| G1 | All four witnesses produce **legacy-identical verdicts** under the hop-off |
| G2 | **gap-2 does not return** — the corpus-wide divergence from clang's promotions over the OM bodies that killed the general relational mirror. This is the gate most likely to fail; run it before believing G1 |
| G3 | The coupled scope's anti-masking gate still holds: `neural-net_fail` (`--fixedbv`) reports FAILED. A widened conversion rule is exactly the shape that masked it once |
| G4 | Default path unaffected — the flag is default-off; assert with a default-path slice anyway |
| G5 | Dual-solver agreement (Bitwuzla + Z3) |

Census methodology is inherited verbatim from the coupled scope's §5 (skip
tests that already pass the flag; count both-paths-no-verdict separately;
exclude `--k-induction-parallel`; 200-byte minimum; sample dense and unbiased).
Add the rule the `frontends-to-irep2.md` Phase 1 work established: **probe the
invariant you depend on and prove the probe fires on known-bad input before
trusting a zero.**

## 8. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | The narrowed rule still re-opens gap-2 | G2, run first; if it cannot be threaded, record that and leave the four tests pinned rather than shipping a corpus-wide regression |
| R2 | A widened conversion masks a real bug, as the assignment arm did | G3 — the `neural-net_fail` anti-masking gate is not optional |
| R3 | §4 is wrong and the four tests share no mechanism after all | Phase 0 is cheap and answers exactly this; a refutation still advances the flip by eliminating a hypothesis |
| R4 | Fixing the comparison gap exposes a further seam (the return seam §10.3 already flags for `--fixedbv`) | Expected; record it rather than widening scope mid-flight |

## 9. Non-goals

- The array-typed assignment (`scope-array-assignment-conversion.md`).
- The coupled scope's own gates — discharged, see its §13.1.
- `frontends-to-irep2.md` Phase 1 — that is the shared `goto_convert`
  dispatcher, a different pass.

## 10. One-line summary

Four tests that survived every arm the coupled scope built share one property —
an integer and a floating-point operand meeting at a comparison or a bitwise
assignment — and `python_adjust`'s relational arm admits only `bv`/`bv` pairs
while its arithmetic arm admits floatbv too; the fix is a narrowed widening of
that asymmetry, gated hard on the corpus-wide divergence that killed the
general version.
