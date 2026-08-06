# Scope — relational/equality reconciliation over mixed float and integer operands

> **Status: Phase 0 run 2026-08-03 — §4 confirmed for three of the four
> witnesses, refuted for `sum_tuple`, which needs re-homing (§11); the
> traffic-volume half is partially answered and lowers the gap-2 prior (§12);
> **REOPENED (§17): §16's closure was wrong. `lambda15` passes with two arms
> (equality-with-bool + ordering-with-float); it had two defects queued, and
> every earlier attempt fixed at most one. The widening still needs narrowing
> to §12's measured traffic before it is shippable.**
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

## 11. Phase 0 — results (2026-08-03): §4 is confirmed for three of four

§4 was recorded as a hypothesis with instructions not to skip Phase 0. Phase 0
was run, and it **partially refutes** the hypothesis. That changes the scope.

### 11.1 Method

The comparison dispatch in `python_adjust::adjust_expr` was widened to *observe*
`equality2t`/`notequal2t` alongside the four ordering relations (observation
only — equality still falls through unadjusted) and instrumented to log, per
node: the kind, both operand type kinds, whether either side is floating point,
whether both are `bv`, and whether the existing guard would admit it. Each
witness was run under `--python-irep2-adjust-only --goto-functions-only` with
its own `test.desc` flags. The instrumentation was reverted.

### 11.2 Result

**Every observed comparison has `admitted=0`** — across all four witnesses the
existing relational guard never fires once. The homogeneous traffic
(`signedbv signedbv`, `unsignedbv unsignedbv`, `floatbv floatbv`) is
operational-model background and is identical across the four, so it is not
what distinguishes them.

The heterogeneous pairs — the ones §4 predicts — are:

| witness | heterogeneous comparison pairs | §4 holds? |
|---|---|---|
| `chained-comparison2_fail` | `lessthan floatbv signedbv`, `lessthanequal floatbv signedbv`, `lessthanequal signedbv floatbv` | **yes** |
| `lambda15` | `lessthanequal floatbv signedbv`, `equality floatbv bool` | **yes** |
| `precedence2` | `equality floatbv signedbv` — an **equality**, which has no arm at all | **yes**, via the second half of §4 |
| `sum_tuple` | **none** | **no** |

### 11.3 What this changes

1. **§4 is confirmed for three of the four.** A float/integer comparison
   reaches the dispatch and no arm admits it. `precedence2`'s is an *equality*,
   so the missing `equality2t`/`notequal2t` arm is load-bearing on its own and
   not merely a tidiness gap — this scope must cover both halves.
2. **`sum_tuple` is refuted and must be re-homed.** It contains no
   heterogeneous comparison at all, so whatever it hits is a third mechanism,
   distinct from both this scope and the array-typed assignment. It should be
   removed from this scope's witness list rather than carried as a straggler —
   the coupled scope's §7 made exactly that mistake in the other direction.
3. **The `admitted=0`-everywhere result narrows Phase 1.** The existing
   signedness-mismatch guard is inert on this traffic, so widening it cannot
   regress these tests by changing what it already does; the risk is entirely
   in what the *new* admission rule lets through. G2 (gap-2) remains the gate
   that matters.

### 11.4 Corrections to this document

- §2's table lists `sum_tuple` as sharing the mixed-kind property. That is
  **wrong** on measurement; the row should read "no heterogeneous comparison —
  re-home".
- §6 Phase 0's question 2 (how much corpus traffic a widened rule would touch)
  is **not** answered here: only the four witnesses were censused, not a corpus
  slice. Phase 1 still needs that number before the admission rule is chosen,
  and G2 cannot be discharged without it.

## 12. Phase 0 part 2 — the traffic-volume question (2026-08-03)

§11.4 recorded that the corpus-slice half of Phase 0 was unanswered and that G2
could not be discharged without it. Partially answered here.

### 12.1 Method and sample

The same instrumentation, reduced to two booleans per comparison node —
`mixed` (one operand floatbv, the other bv) and `samekind_widthdiff` (both bv,
same signedness, different widths, the shape gap-2 says must stay untouched) —
run over an unbiased stride slice of `regression/python` under
`--python-irep2-adjust-only --goto-functions-only`, plus a trivial
`assert True` program as the operational-model baseline.

**Sample: 14 corpus tests and the baseline.** That is small, and the reason is
recorded rather than hidden: each Python test spawns the parser subprocess, and
the dev machine was contended for most of this session. The slice was unbiased
(stride over the directory listing), not a prefix.

### 12.2 Result

| population | comparison nodes | `mixed=1` | `samekind_widthdiff=1` |
|---|---:|---:|---:|
| OM baseline (`assert True`) | 1 296 | **0** | **0** |
| 14-test corpus slice | ~18 700 | **0** | **0** |
| the three confirmed witnesses (§11.2) | — | present | — |

### 12.3 What it means for G2

**The gap-2 risk to a float/integer admission rule looks low, on this sample.**
gap-2 killed the *general* relational mirror because it diverged over the
operational-model bodies. Those bodies produce **1 296 comparison nodes and not
one mixed pair** — the OM traffic a float/int rule would newly admit is, on this
evidence, empty. The mixed pairs are concentrated in the witnesses.

Equally, the `samekind_widthdiff` count is zero everywhere, so the char-vs-int
promotions gap-2 protects are not reached by this dispatch at all in the sample
— which suggests the narrowed rule §5 asks for may not even need a separate
exclusion clause.

### 12.4 What this does *not* discharge

**G2 is not discharged.** A zero over 14 tests bounds the *frequency* of the
traffic, not the *behaviour* of a widened rule on the tests that do have it. G2
remains a verdict-parity gate over a proper corpus run and must be executed
against a real implementation, on a machine that can complete it. What §12 buys
is a prior: the rule is unlikely to be corpus-wide destructive, so Phase 1 is
worth attempting rather than being blocked on the fear that killed gap-2.

## 13. Phase 1 attempt 1 — refuted (2026-08-03)

Phase 1 was attempted twice and **neither widening clears the witnesses**. Both
attempts were reverted. This section records what they rule out, because the
result redirects the phase.

### 13.1 What was tried

| # | change | result on `lambda15` under the hop-off |
|---|---|---|
| 1 | admit a float/integer pair at the relational dispatch, and route `equality2t`/`notequal2t` through it for that case | still aborts |
| 2 | additionally reconcile `equality`/`notequal` whenever both operands are numeric and their types differ | still aborts |

The abort in both cases:

```
Assertion failed: (a->sort->get_data_width() == b->sort->get_data_width()),
  function mk_eq, file bitwuzla_conv.cpp, line 470
```

### 13.2 What it rules out

The failing node is an **equality between two operands of different bit
widths** that reaches the solver unreconciled. Attempt 2 reconciles exactly
that shape at `python_adjust`'s comparison dispatch and the abort survives, so:

> The mismatched equality **does not pass through
> `python_adjust::adjust_expr`'s comparison arm** — it is either created after
> the adjuster runs, or it lives somewhere the walk does not visit.

Widening this dispatch further is therefore the wrong move, and §5's framing —
"the fix is a narrowed widening of the relational asymmetry" — is **too narrow
a diagnosis** for at least `lambda15`.

### 13.3 Correction to §11's method

§11 classified witness operands by *kind* (`floatbv`, `signedbv`, …) and found
heterogeneous pairs in three witnesses. That classification **cannot see a
same-kind, different-width pair** — two `signedbv` operands of 32 and 64 bits
report identical kinds. The `mk_eq` abort is exactly that shape, so §11's
"confirmed for three of four" understates what those tests contain. §12's
corpus census used a `samekind_widthdiff` flag and correctly returned zero for
the corpus, but it was never run against the witnesses themselves.

**Re-run the §12 flags over the four witnesses before Phase 1 attempt 2.** That
is the measurement this phase actually needs, and it is cheap.

### 13.4 Revised Phase 1 entry condition

1. Measure `samekind_widthdiff` on the witnesses (§13.3).
2. Locate where the mismatched equality is *built* — converter, a list/dict
   model, or a post-adjust pass — rather than assuming the adjuster can see it.
3. Only then choose the layer, as §5 requires.

## 14. §13.2 measured, and the scope's premise is wrong (2026-08-03)

§13.4 asked for the width measurement over the witnesses before any further
attempt. It was run, and it settles the question against this scope.

### 14.1 The measurement

The comparison dispatch was instrumented to log any node whose two operands
have different bit widths, and each witness run under
`--python-irep2-adjust-only --goto-functions-only` with its own flags:

| witness | width-mismatched comparisons reaching `adjust_expr` |
|---|---|
| `lambda15` | **0** |
| `chained-comparison2_fail` | **0** |
| `precedence2` | **0** |
| `sum_tuple` | **0** |

Not one, in any of them — while the same dispatch sees ~1 300 comparison nodes
per run, so the walk is reaching the traffic. §13.2's inference is now a
measurement.

### 14.2 What that means for this scope

**The equality that aborts `mk_eq` is not something `python_adjust` can fix.**
It is not in the comparison traffic the adjuster walks, so no admission rule at
that dispatch — narrow, wide, or unconditional — can reconcile it. §5's
premise ("the fix is a narrowed widening of the relational asymmetry") is
refuted for the shape that actually aborts.

The abort surfaces during *"Encoding remaining VCC(s)"*, i.e. after symex, which
leaves two candidates:

1. the node is in the GOTO but outside what `adjust_expr` walks, or
2. it is synthesised during symex — renaming, dereferencing or a container
   model's index arithmetic.

The census used `--goto-functions-only` and so cannot distinguish them: it stops
before symex runs. That is the next probe, and it belongs at the symex/SMT
boundary, not in the frontend.

### 14.3 Status of the four witnesses

| witness | standing |
|---|---|
| `sum_tuple` | refuted in §11 — no heterogeneous comparison at all |
| `lambda15` | aborts in `mk_eq`; **not reachable from this dispatch** |
| `chained-comparison2_fail`, `precedence2` | do contain float/int comparisons this dispatch sees (§11), but whether reconciling them clears the tests is **untested** — both attempts aborted first on `lambda15`'s shape |

**This scope should not proceed to Phase 1 as written.** The honest next step is
to re-triage the group at the symex/SMT layer and decide whether it is one
mechanism or two: an adjuster-visible float/int gap for two witnesses, and a
downstream width gap for `lambda15`. Re-scoping is cheaper than another
widening attempt, and §5's option table should be redrawn once the layer is
known.

## 15. The aborting node identified — and §14 corrected (2026-08-04)

§14 concluded the failing equality never reaches `python_adjust`. **That
conclusion was wrong, and the reason is a bug in its own probe.**

### 15.1 The node

Instrumenting `smt_convt`'s `equality_id` case (`smt_solver.cpp:1320`) to report
operand types whenever the converted sorts disagree, `lambda15` yields exactly
one line before the abort:

```
EQPROBE  w=64/1  s1=floatbv  s2=bool
```

The aborting node is an equality between a **64-bit floatbv and a 1-bit bool** —
`assert is_even(4) is True`, where the lambda's result is monomorphised to
float and compared against a Boolean literal.

### 15.2 Why three probes and two fixes all missed it

`is_bv_type` (`irep2_utils.h:13`) covers `unsignedbv_id` and `signedbv_id`
**only**. Bool is not a bv type. Every artefact in §11-§14 keyed off that
predicate:

| artefact | guard | effect on a `bool` operand |
|---|---|---|
| §14's width probe | `w = (is_bv_type \|\| is_floatbv_type) ? width : 0`, logged only if `w0 && w1` | bool → 0 → **never logged**, hence the false "0 mismatches" |
| §13 attempt 1 | `float_int_mix` = floatbv paired with `is_bv_type` | bool excluded |
| §13 attempt 2 | `numeric` = `is_bv_type \|\| is_floatbv_type` | bool excluded |

One predicate produced three independent false negatives. **§14's central claim
— that the node is invisible to the dispatch — is withdrawn**: §11's own kind
census had already logged `equality floatbv bool` for `lambda15`, and that
evidence was mis-filed as an ordinary heterogeneous pair instead of the
aborting shape.

### 15.3 A third attempt, also refuted

Admitting `bool`-paired-with-numeric for `equality`/`notequal` at the dispatch
and calling `c_implicit_typecast_arithmetic` **does not clear the abort**;
`lambda15` fails identically. So either the arm does not fire for this node, or
the helper does not convert a Boolean operand, or the equality that reaches the
solver is rebuilt after the adjuster runs. Reverted, not shipped.

### 15.4 The next measurement, stated so it is not guessed again

Do **not** attempt a fourth widening. Determine, in order:

1. Does the arm fire for this node? Probe inside the guard, print on entry —
   the discipline `frontends-to-irep2.md` §11.4 already requires and that this
   investigation kept skipping.
2. If it fires, does `c_implicit_typecast_arithmetic` change the operand types?
   `get_c_type` maps bool to `BOOL` (`c_typecast.cpp:396`), which may rank below
   the float and leave the pair unreconciled.
3. If both hold, the node reaching the solver is a *different* instance —
   rebuilt during symex — and the fix does not belong in the frontend at all.

Each is a one-line probe. Three attempts were spent on hypotheses that a probe
would have killed in minutes.

## 16. Resolved: the fix is not in the frontend (2026-08-04)

§15.4 listed three probes and forbade a fourth widening before they were run.
They were run. The answer is conclusive and closes this scope.

### 16.1 The three probes

| # | question | answer |
|---|---|---|
| 1 | Does the arm fire for the `floatbv == bool` node? | **Yes** — 6 times in `lambda15` |
| 2 | Does `c_implicit_typecast_arithmetic` convert it? | **It did not, and that was a real bug** — the IREP2 `get_c_type` overload never classified `floatbv`, so it returned `OTHER`, outranking every arithmetic kind, and the helper converted neither operand. Fixed in **PR #6688** |
| 3 | With the helper fixed, does the arm reconcile the node? | **Yes** — `before=floatbv/bool` → `after=floatbv/floatbv` |

**And `lambda15` still aborts in `mk_eq`.**

### 16.2 The conclusion

Arm fires, helper converts, operands reconciled, abort unchanged. Therefore the
equality that reaches the solver is **a different instance**, rebuilt after
`python_adjust` runs. No admission rule at this dispatch can fix it — which is
what §14 claimed for the wrong reason and §15 withdrew for a good one. The
claim is now true and *measured*, not inferred.

**This scope is closed.** The mechanism is downstream of the frontend, so it
belongs to whoever owns the symex/SMT boundary, not to a `python_adjust`
scope. The equality arm itself was implemented and works, but clears no test,
so it was **not shipped**: a behaviour change to a flag-gated path with no
demonstrated benefit is not worth its risk.

### 16.3 What this investigation produced

| output | status |
|---|---|
| **PR #6688** — `get_c_type` never classified `floatbv`, making `c_implicit_typecast_arithmetic` a silent no-op for five Python call sites, four on the **default** path | shipped |
| `sum_tuple` refuted as a witness (§11) | recorded |
| The aborting node identified as `floatbv == bool` (§15) | recorded |
| The mechanism located downstream of the frontend (§16) | recorded |

The scope's original hypothesis was wrong; chasing it found a real latent bug in
a shared helper that nothing else had surfaced.

### 16.4 For whoever takes the downstream mechanism

Start at `smt_convt`'s `equality_id` case (`smt_solver.cpp:1320`) — probing
there names the node in one build. Do **not** start in the frontend: four
attempts there are recorded above, and all four are refuted.

## 17. §16 is wrong — `lambda15` *is* frontend-fixable (2026-08-04)

§16 closed this scope on the conclusion that the aborting equality is "a
different instance, rebuilt after `python_adjust`", so no admission rule at that
dispatch could fix it. **That conclusion is refuted by measurement.**
`lambda15` now passes under `--python-irep2-adjust-only`.

### 17.1 What §16 got wrong

§16's evidence was: the arm fires, the helper converts, the abort is unchanged.
All three observations were correct. The inference was not — because
**`lambda15` has two independent defects, and every attempt fixed at most one**:

| # | node | needs |
|---|---|---|
| 1 | `floatbv == bool` (`mk_eq` sort-width abort) | an **equality** arm admitting a Boolean paired with a number |
| 2 | `lessthanequal` with mixed operands (`convert_ast_node` signedbv assert) | the **ordering** arm admitting a float/integer pair |

Fixing only #1 leaves #2 aborting, and vice versa. Every earlier attempt
(§13.1's two, §15.3's third) used `is_bv_type` in its guard, which **excludes
bool** (§15.2) — so none of them cleared #1, and the run never reached #2. The
abort never moved, which read as "the fix does not apply" and was actually
"a second defect is queued behind the first".

The tell was available and missed: when the equality arm finally admitted bool,
**the abort message changed** — from `mk_eq` to `convert_ast_node:1563`. A
changed failure is progress; an unchanged one is not. That distinction is worth
checking explicitly whenever a fix "does nothing".

### 17.2 Measurement

With both arms present:

| witness | expected | hop-off |
|---|---|---|
| `lambda15` | SUCCESSFUL | **SUCCESSFUL** |
| `chained-comparison2_fail` | FAILED | still aborts |
| `precedence2` | SUCCESSFUL | FAILED |
| `sum_tuple` | SUCCESSFUL | still aborts |

`neural-net_fail` (`--fixedbv`) still reports **FAILED**, so the anti-masking
gate holds against both arms.

### 17.3 Status, honestly

One of four witnesses clears. §4's original hypothesis — that the relational
arm's `bv`-only admission is implicated — is **partly vindicated**, having been
prematurely written off in §14 and §16. The scope should be **reopened**, not
closed.

**What is not yet done, and must be before any of this ships:** the ordering-arm
widening used to obtain this result is deliberately broad (any two differing
numeric types), which is precisely the shape gap-2 rejected for diverging
corpus-wide over the operational-model bodies. §12's census says the *mixed
float/integer* traffic is empty in the OM bodies, but that census did not cover
the same-kind-different-width traffic this broad rule also admits. **The rule
must be narrowed to what §12 actually measured, and G2 re-run, before it is a
patch rather than an experiment.** Nothing from §17 has been shipped.

## 18. §17.3 shipped; the residue re-measured, and `precedence2` root-caused (2026-08-06)

§17.3's closing line — "Nothing from §17 has been shipped" — has been stale
since the next day. The narrowing it demanded was implemented and merged as
**PR #6702** (`9fdd6d7e45`, 2026-08-05): the ordering arm admits a
`float_int_mix` pair rather than "any two differing numeric types", and the
equality arm is restricted to a Boolean paired with a number. What #6702 left
open was its own **outstanding gate — the full verdict-parity sweep**, which
had timed out repeatedly on a contended machine. That sweep is §18.4.

### 18.1 The witnesses, re-measured

Fresh binary, confirmed to carry `9fdd6d7e45` by a positive probe rather than
an mtime (`lambda15` clears only with the arm present).

| witness | legacy | hop-off | | vs §17.2 |
|---|---|---|---|---|
| `lambda15` | SUCCESSFUL | SUCCESSFUL | **SAME** | unchanged |
| `chained-comparison2_fail` | FAILED | FAILED | **SAME** | **newly clears** — §17.2 had it aborting |
| `precedence2` | SUCCESSFUL | FAILED | DIVERGE | unchanged |
| `sum_tuple` | SUCCESSFUL | FAILED | DIVERGE | **was aborting; now reaches a verdict** |
| `neural-net_fail` (`--fixedbv`) | FAILED | FAILED | **SAME** | G2 anti-masking holds |

Two of the four §9.4 witnesses now agree, not one. `chained-comparison2_fail`
is `assert 0 <= x <= 2 < 3` with `x = 2.5` — precisely the mixed float/integer
ordering shape `float_int_mix` admits — so #6702 owns that flip.

The two residuals also **changed kind**: both now produce a verdict where they
previously aborted, and both diverge in the **false-alarm** direction (hop-off
FAILED where legacy is SUCCESSFUL), not the masking direction. That is the
cheaper failure to carry, and per §17.1 a changed failure is progress.

### 18.2 `precedence2` is an assignment conversion, not a comparison

A GOTO diff of `python_user_main` — the two dumps are otherwise instruction-for-
instruction identical — isolates one shape, at `x &= 6` / `x |= 7` where `x`
carries `double`:

```
legacy:   ASSIGN x = (double)((signed long int)((signed int)x) & 6);
hop-off:  ASSIGN x =          (signed long int)((signed int)x) & 6;
```

Both arms keep the bitwise operation integral, so the arithmetic arm is not
implicated. The whole difference is the cast **at the assignment seam**: the
hop-off stores an integer into a `double` lvalue, and `assert x == 7` (line 76)
then fails. The counterexample names line 76, which is why the earlier readings
looked like a comparison defect — the violated assertion is three statements
downstream of the assignment that corrupts `x`.

### 18.3 Why the assignment arm declines it — a `floatbv` gap in `c_typecast`

The general assignment arm (`python_adjust.cpp:617-655`) admits this: both
sides are numeric, and the types differ. It calls `c_implicit_typecast`, which
**silently declines**, leaving the source unchanged.

The cause is in the shared helper. `check_c_implicit_typecast`'s two overloads
do not agree:

| source kind | `typet` overload | `type2tc` overload |
|---|---|---|
| `bool` → `floatbv` | permitted | **absent** |
| `bv` → `floatbv` | permitted | **absent** |
| `floatbv` → anything | permitted | **no `floatbv` source branch at all** |

`c_typecast.cpp:195-279` has no `floatbv` case anywhere — not as a destination,
not as a source — so every implicit conversion touching a float falls through
to `return true` (reject). Since ESBMC represents Python floats as `floatbv`
and reaches `fixedbv` only under `--fixedbv`, this rejects the default float
representation entirely.

This is the same defect family as **PR #6688**, which found `get_c_type` never
classifying `floatbv` and thereby making `c_implicit_typecast_arithmetic` a
silent no-op (§16.3). The IREP2 port of this file omitted floats in more than
one place.

**Blast radius, stated because the fix leaves this scope's flag-gated area.**
The `type2tc` overload has exactly two callers:

| caller | path |
|---|---|
| `python_adjust.cpp:652` | `--python-irep2-adjust[-only]`, default-off |
| `interval_domain.cpp:820` | **default path** under interval analysis |

So the fix is not free the way the arms in §17 were: it changes a shared helper
whose second caller is reachable without any experimental flag, and it needs
gates of its own rather than this scope's witness table.

### 18.4 The outstanding gate — census methodology and status

#6702's sweep is being re-run as a **whole-corpus** census over all 4 509
`regression/python` tests, two runs each. It doubles as **G4** of
`scope-coupled-arith-assign-conversion.md` §5, so it adopts that section's five
inherited, non-optional rules. Rule 1 is not hypothetical here: **56 tests
already pass `--python-irep2-adjust*` in their own `test.desc`** and would have
received the flag twice, which makes boost throw `multiple_occurrences` — 56
false divergences, against the 9 that invalidated the first census on this
track. A first, non-compliant run of this sweep was discarded for exactly that
reason before any result was read from it.

Partial result at 901/4 509 (the run is in flight; the full table lands with
the next revision of this section):

| bucket | count |
|---|---:|
| SAME | 892 |
| DIVERGE | 5 |
| NOVERDICT_BOTH (rule 2, not attributable) | 1 |
| SKIP_SHORT (rule 4) | 3 |

The five divergences split into two kinds, and only one is established:

| test | legacy | hop-off | status |
|---|---|---|---|
| `class10`, `class12` | SUCCESSFUL | `rc=134` | **real** — root-caused to an array-into-scalar assignment, `scope-array-assignment-conversion.md` §13 |
| `del_list_slice`, `dictcomp_over_items` | SUCCESSFUL | TIMEOUT | **unconfirmed** |
| `dict24_fail` | FAILED | TIMEOUT | **unconfirmed** |

**The three timeouts must not be reported as divergences without a serial
re-run.** They were measured with 20 concurrent workers on a machine that also
carried other work, and a hop-off-only timeout is exactly the artifact that
contention produces; the harness's 130 s cap is per-run, not per-arm. They are
recorded here as *candidates*, and the rule-2 discipline that keeps
both-paths-no-verdict out of the attributable count applies with equal force to
a one-sided timeout under load.

### 18.5 Status

The scope's §17 work is **shipped**; its gate is **in flight**; and its larger
residual now has a named root cause in a shared utility rather than in the
comparison dispatch this document spent §13-§16 searching. `sum_tuple` remains
unexplained by §18.3 — its counterexample shows unresolved `tuple_elem`
symbols, consistent with §11's finding that it has no heterogeneous comparison
at all, and it should not be assumed to share `precedence2`'s cause.

## 19. §18.3's fix, shipped and measured (2026-08-06)

The `floatbv` gap §18.3 named is fixed: `check_c_implicit_typecast`'s `type2tc`
overload now admits `floatbv` as a destination in the Boolean and integer source
branches, and carries a `floatbv` source branch of its own, mirroring the
`typet` overload it was ported from.

### 19.1 What it clears

Measured A/B against a frozen pre-fix binary, so each row is a controlled
comparison rather than a recollection of an earlier run:

| test | legacy | hop-off before | hop-off after |
|---|---|---|---|
| `precedence2` | SUCCESSFUL | FAILED | **SUCCESSFUL** |
| `math33_frexp` | SUCCESSFUL | FAILED | **SUCCESSFUL** |
| `math_edge_frexp_success` | SUCCESSFUL | FAILED | **SUCCESSFUL** |

The two `frexp` tests were **not** predicted by §18.3 — they came out of the
census and were found to share the cause. That is the whole argument for
running a census rather than fixing named witnesses: one shared-helper defect
was producing divergences in three unrelated tests, and only one of them was on
the witness list this scope had been working from since §4.

### 19.2 Gates

| gate | result |
|---|---|
| G2 anti-masking (`neural-net_fail --fixedbv`) | **FAILED**, unchanged |
| `lambda15` | SUCCESSFUL, unchanged |
| unit suite | 635/635 (632 pre-existing + 3 new) |
| `floats` + `floats-regression` | 164/164 |
| `--interval-analysis` corpus — the **default-path** caller at `interval_domain.cpp:820` | 111/112 |

The single interval failure, `esbmc-unix/github_2513_1`, is a **pre-existing
timeout**, not a regression: it takes **138.3 s with the fix and 147.8 s
without** it against the same 120 s harness cap, and produces the expected
`VERIFICATION FAILED` verdict when run directly. Timing both arms is what
settles this class of failure; the ctest verdict alone cannot.

### 19.3 The regression test, and why it is a unit test

Five attempts at a minimal Python reproducer all failed to discriminate: the
shape that exercises the seam is an integer-typed expression assigned into a
`double` lvalue, and outside `precedence2`'s whole-module type inference the
frontend simply re-types the variable and the conversion is never needed. The
one reduction that *did* discriminate — a compound `x &= 6` over a float — is
**rejected by CPython**, so it would fail `scripts/check_python_tests.sh` and is
not a legitimate test input.

The defect's unit is the helper, so `unit/util/c_typecast.test.cpp` pins it
directly: the float admissions, a struct-source control that must stay
rejected, and the behavioural case that the cast is actually inserted.
Mutation-checked — with the fix reverted, 6 of 7 assertions fail and the
control still passes, so the control is not vacuous.

### 19.4 Census status

At 3 757/4 509: 3 677 SAME, 18 DIVERGE, 56 SKIP_PREFLAGGED (rule 1 earning its
place), 4 SKIP_SHORT, 2 not attributable. The 18 divergences, all measured on
the **pre-fix** baseline:

| class | count | status |
|---|---:|---|
| cleared by §19's fix | 3 | done |
| `rc=134` abort | 7 | `class10`/`class12` root-caused (`scope-array-assignment-conversion.md` §13); `github_3866`×3, `missing-return14_fail`, `min_max_multi_args` unexamined |
| `rc=139` segfault | 2 | `github_3658_7`, `return13-fail` — unexamined |
| hop-off-only TIMEOUT | 6 | **unconfirmed**, see §18.4 — needs a serial re-run |

G4 is not discharged: it requires 0 attributable divergences, and there are at
least 9 real ones outstanding.
