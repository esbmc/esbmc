# Spike — IREP2-native value-operand location carriage (the W1-loc keystone)

**Program:** repo-wide "IREP2-native frontend→goto pipeline" (the Part V umbrella, #4715).
**This issue:** the mandatory first spike. Read-only investigation + one throw-away prototype.
**Owner:** TBD. **Status:** Phases A+B executed (2026-07-05) — **D3 selected (go),
D1 ruled out; W1-loc refuted as a *fidelity* wall** (the barrier is native-
dispatcher implementation cost, not location correctness). Phase C prototype is
the next step. See Appendices A & B. **Refs:** #4715, Part V of
`docs/irep2-migration.md`.

---

## 1. Why this is the keystone (and why it is *not* V.1k/W2)

The §V.6 recommendation named "Phase V.1k (resolve-then-build)" as the mandatory
first spike. That was correct when written, but the V.1k *breakthrough* (doc
§"V.1k breakthrough") has since **discharged W2**: the converter builds
`member2t`/`index2t` over symbol-typed bases via the relaxed construction assert
+ exact IREP2 round-trip, **0 regressions, 536/536** on an asserts build.
Resolution stays exactly where it is (`clang_cpp_adjust`); nothing in
expression construction is blocked on resolve-then-build any more.

The binding constraint for **literal 100%** — the §V.1 acceptance bar, where the
function **body** handed to `goto_convert` is IREP2 with **no** `migrate_*`
back-hop — is now a single, different wall: **W1-loc**.

> **W1-loc (verified, `goto_convert_functions.cpp:110-120`):** IREP2 value-level
> expressions carry **no source location**. Only the structured-CF code kinds got
> the V.4.1/V.4.5 non-reflected `location` field (`irep2_expr.h:779`, `:1090`,
> `:1148`, `code_block2t::end_location` `:1842`). The clang frontends stamp every
> sub-expression of a statement with that statement's `#location`; the
> legacy→IREP2→legacy body round-trip **drops it from value operands**.
> `goto_convert` then generates instructions from those operands (the tmp/GOTO
> sequence a `&&`/`||` short-circuit or in-expression call lowers to, location
> read at **`goto_sideeffects.cpp:242`**) with an **empty** location — breaking
> any instruction-location-keyed pass (`--condition-coverage`, counterexample
> text, witness columns).

The current mechanism that hides this is `restore_value_locations`
(`goto_convert_functions.cpp:124`, called at `:187`): after the round-trip it
walks the **legacy** body and re-stamps each value operand with the **inherited
statement location**. That round-trip-to-legacy is therefore *load-bearing*, and
it is the literal "legacy-at-the-seam" the 100% bar forbids. The V.4 outcome
correctly classified W1 as a `RETAIN_BOUNDARY` **under the then-available
infrastructure**; this spike asks whether new, narrowly-scoped infrastructure
removes it cheaply.

**Acceptance bar for the whole program (unchanged, repo-wide — doc §V.1):**
1. `git grep -P '\b([A-Za-z_]*(exprt|typet|codet)|irept)\b' -- src/<frontend>` → ~0 (enumerated glue only).
2. Symbol table written only via `set_type(type2tc)`/`set_value(expr2tc)` carrying fully-IREP2 values.
3. Bodies are IREP2; `goto_convert` consumes them with **no** `migrate_*` back-hop (today: `goto_convert_functions.cpp` round-trip + `restore_value_locations`).
4. No `#`-attribute legacy-`irept` escape hatch survives into a shared downstream pass.

This spike settles only the precondition for (3). W2/W3/W4 are separate, already-scoped, and do **not** gate it.

---

## 2. The decisive empirical question

`restore_value_locations` re-stamps operands with the **enclosing statement's
location**, not with any per-operand-distinct location. If that is faithful —
i.e. the location a frontend originally put on each value operand *equals* the
enclosing statement's `#location` for every operand `goto_sideeffects` reads —
then **no value node needs to store a location at all**: an IREP2-native pass can
inherit the statement location at consumption time and reproduce today's bytes.

> **Q-LOC (the keystone question):** Across the corpus, is the per-operand source
> location that today's pipeline consumes (at `goto_sideeffects.cpp:242` and
> every other operand-location read) **always equal** to the enclosing
> statement's location? Or do genuine per-operand-distinct locations exist (e.g.
> a multi-line expression where a sub-call sits on a different line than the
> statement head)?

The answer selects the design and sets the cost. The frontend comment ("stamp
every sub-expression of a statement with that statement's `#location`") is
*evidence for* equality but is not a proof — multi-line expressions and macro
expansions are the suspected counterexamples and must be measured, not assumed.

---

## 3. Design options

| | Mechanism | Cost | Sound iff |
|---|---|---|---|
| **D3 — inherited-location pass** *(lead if Q-LOC holds)* | Port `restore_value_locations`'s inheritance into the IREP2-native `goto_convert`/`remove_sideeffects`: thread the enclosing `code_*2t::location` down to operands **at consumption time**. No node stores a location. No round-trip. | Smallest — one pass, mirrors existing logic. | Q-LOC = "always equal" (or the exceptions are characterizable and reproducible by rule). |
| **D2 — located side-effect operands** *(fallback)* | Add the non-reflected `location` field **only** to the operand kinds `goto_sideeffects` actually reads locations from (the side-effect-bearing set: in-expression `function_call`, `++`/`--`, comma, and the `&&`/`||`/`?:` short-circuit operands — note ternary/unary/constant macros already carry it). Census-bounded, not every node. | Medium — touches a bounded node set + the builders that emit them in every frontend. | The side-effect-operand set is the *complete* set of operand-location reads (prove by census). |
| **D1 — location on `expr2t` base** *(rejected baseline, re-cost)* | Non-reflected `location` on every value node, excluded from hashing via the **existing** `excluded_field_bytes` pattern (`irep2_expr.h:781`). | Largest — memory on every node; every builder/migrate path must carry it. | Hashing/equality provably ignore location (dedup/CSE unaffected). |

The V.4 outcome rejected D1 as "a large, separate initiative." The new
observation is that **D3 may make the whole wall cheap**, because
`restore_value_locations` is itself proof that inheritance — not storage — is the
operative semantics today. The spike's job is to confirm or refute that.

---

## 4. Spike plan (one throw-away branch, do not merge)

**Phase A — Census (read-only).** Enumerate *every* site that reads a location
off a value operand during goto-conversion / side-effect removal, not just
`goto_sideeffects.cpp:242`. Grep `goto_sideeffects.cpp`, `goto_convert*.cpp`,
`remove_sideeffects` paths for `.location()`/`location()` on operands. Output:
the complete operand-location-read set (this both bounds D2 and tells D3 what to
thread).

**Phase B — Settle Q-LOC (empirical).** Instrument `restore_value_locations` to
log, per operand, the operand's **pre-round-trip** location (captured before the
body round-trip drops it) vs. the **inherited statement** location it would be
re-stamped with. Run over `regression/esbmc` + `regression/python` +
`regression/cbmc`. Tabulate: % operands where they are equal; characterize every
inequality (multi-line expr? macro? generated temp?). This is the go/no-go input.

**Phase C — Prototype the leading design on ONE frontend (Python).** Build the
IREP2-native consumption path so the Python body reaches `goto_convert` as IREP2
with **no** round-trip and **no** `restore_value_locations`. If Q-LOC holds:
D3 (inheritance pass). Else: D2 (located operands for the inequality set).
Feature-flag it (`--irep2-native-body`, off by default), Python-only.

**Phase D — Measure against the proof obligation (§5).** Diff GOTO output
byte-for-byte (including locations) flag-on vs. flag-off across the Python
corpus + the operational-model `.py` corpus + an `esbmc-cpp` smoke stratum (the
pass is shared). Report.

Throw-away branch named `spike/w1loc-location-carriage`, **not merged** —
deliverable is the report + go/no-go, exactly as the V.1k rounds 1–3 were run.

---

## 5. Proof obligations (non-negotiable acceptance gates)

A design passes the spike only if, with the flag on:
1. **Byte-identical GOTO including locations** vs. today, over the Python +
   model-`.py` corpora. (Locations are the whole point; a location diff is a
   failure, not a nit.)
2. **`--condition-coverage` parity** — the named failure mode in
   `goto_convert_functions.cpp:118` must not regress.
3. **Dual-solver verdict parity** (Bitwuzla + Z3) + **counterexample-text
   parity** where `test.desc` pins it.
4. **Hashing/dedup unaffected** — prove location does not enter `crc`/`do_crc`/
   equality (D1/D2 only): the `excluded_field_bytes` invariant holds; CSE and
   value-set dedup produce identical node counts.
5. **`esbmc-cpp` (+ a Solidity/CUDA stratum) green** on the shared pass — this is
   not Python-private code.

If any gate fails for all three designs, the report ends "W1-loc remains a
`RETAIN_BOUNDARY` — record the measured reason" and 100% is formally
unreachable without location-fidelity degradation (a separate product decision).

---

## 6. Risk register (extends doc §V.4 RV1–RV5)

| # | Risk | Sev | Mitigation |
|---|---|---|---|
| LR1 | Shared blast radius: the IREP2-native goto-convert path changes C/C++/CUDA/Solidity/Java lowering, not just Python (RV1). | **critical** | Feature-flag; one frontend at a time; `esbmc-cpp` gate every commit. |
| LR2 | Q-LOC false — per-operand-distinct locations exist and are not rule-reproducible, forcing D1/D2 with full per-node carriage. | high | Phase B settles this *before* any prototype; D2 census bounds the fallback. |
| LR3 | Location silently enters hashing → dedup/CSE divergence → verdict drift. | high | Gate 4; reuse the proven `excluded_field_bytes` exclusion the CF kinds already use. |
| LR4 | GOTO on-disk format drift (`.goto` binaries) if a node layout changes (D1/D2). | high | Byte-identical GOTO gate; old binaries must still load (Part I B4). |
| LR5 | Scope creep — the spike drifts into W3/V.2 or the V.3 flip. | med | Spike is location-carriage **only**; W2 done, W3/W4 explicitly out (§7). |

---

## 7. Out of scope (these follow the spike; they do not gate it)

- **W2 / V.1k resolve-then-build** — already discharged (relaxed assert + round-trip).
- **W3 / V.2** (IREP2 attribute carriage, `#cpp_type`/`#member_name`/`#cformat`) — separate, already scoped.
- **W4 / V.5** (IREP2-native counterexample printer) — separate, boundary-independent.
- **The Python V.3 converter flip** — the "comparatively mechanical" bulk; runs *after* W1-loc unblocks bodies. It is the smallest, last step (doc §V.1).

---

## 8. Deliverables & estimate

**Deliverables:** (1) the Phase-A operand-location-read census; (2) the Phase-B
Q-LOC measurement table; (3) the throw-away Python prototype behind
`--irep2-native-body`; (4) a go/no-go report naming the chosen design (D3/D2/D1
or RETAIN) with the §5 gate results.

**Estimate:** the spike itself is ~1 focused engineer-iteration (read-only census
+ one instrumented run + one prototype), mirroring the V.1k rounds. The *program*
it unblocks (native goto-convert across all frontends, then per-frontend flips)
remains multi-quarter / multi-engineer / dozens of PRs (RV1) — but this spike is
the cheap, decisive experiment that says whether literal 100% is reachable at
acceptable cost, or whether W1-loc stays a documented boundary.

---

## Appendix A — Phase A census result (executed 2026-07-05)

Deliverable (1). Read-only enumeration of **every site that reads a location off a
value operand** during goto-conversion / side-effect removal, over
`src/goto-programs/goto_convert*.cpp`, `goto_convert_side_effect.cpp`,
`goto_sideeffects.cpp`, `remove_*.cpp`. The set is far smaller than "every value
node", which is the first favourable signal.

**The complete operand-location-read set.**

| # | Site | What it reads | Distinct-per-operand, or statement-inherited? |
|---|---|---|---|
| 1 | `goto_sideeffects.cpp:213` (`&&`/`||` short-circuit lowering; the doc's cited `:242` consumer) | `op.location() = expr.location();` then the generated `IF`/`if_exprt` reads `op.location()` at `:225/:227/:234/:236` | **Inherited by construction** — the operand's own location is *overwritten* with the enclosing expr's before it is ever read. Self-labelled "This is a hack for now." Q-LOC holds trivially here. |
| 2 | `goto_convert.cpp:1255` (`convert_dowhile`) | `code.op0().find_location()` — the do-while **condition** operand | The **only** genuinely per-operand read. But post-round-trip the condition's location is whatever `restore_value_locations` stamped = the inherited statement location; so today's bytes already come from inheritance. |
| — | `goto_sideeffects.cpp:122` | `expr.find_location()` — the **enclosing** side-effect expr (stamps the generated decl/assignment at `:128/:134`) | Statement-level, not a per-operand read. |
| — | `goto_sideeffects.cpp:108` | quantifier expr location | Statement-level (the quantifier node itself). |

Writes that stamp *generated* instructions all copy the **enclosing** expr's
location (`goto_convert.cpp:514/516` `then/else_case().location() = expr.location()`;
the `.location() = expr.location()` family throughout `goto_sideeffects.cpp`) —
i.e. inheritance, never a distinct operand location.

**The decisive mechanism.** `restore_value_locations` / `stamp_value_locations`
(`goto_convert_functions.cpp:99-140`), the pass that supplies operand locations
after the body round-trip, is **pure inheritance**: it pushes each statement's own
`#location` down onto every location-less value operand (`stamp_value_locations`
takes a single `loc` and recurses; `restore_value_locations` lets each nested
statement govern its subtree). So the pipeline *already* sources operand locations
by inheritance, not by preserving distinct per-node locations.

**Q-LOC verdict (in-code, within the goto-convert scope W1-loc touches).** Every
operand location the pipeline consumes to generate an instruction is the enclosing
statement's location — forced (`:213`), inherited (`restore_value_locations`), or
(the one distinct read, `:1255`) resolved to the inherited value post-round-trip.
**Consequence: D3 (inherit at consumption time, no per-node storage) reproduces
today's bytes by construction; D1 (per-node `location` on `expr2t`) is
unnecessary; D2's located-operand set reduces to ≈{do-while condition}, itself
inheritance-satisfiable.** This does not by itself discharge the whole spike —
**Phase B** must still confirm empirically over the corpus that no frontend places
a genuinely-distinct operand location that some read *outside* this census
consumes — but it removes the largest cost driver the V.4 outcome cited (per-node
location carriage) from the table before any prototype is written.

**Next (Phase B).** Instrument `restore_value_locations` to log, per operand, the
pre-round-trip operand location vs. the inherited statement location it re-stamps,
over `regression/{esbmc,python,cbmc}`; tabulate the equality rate and characterise
every inequality (multi-line expr / macro / generated temp). A ~100 % equality
rate promotes D3 from "provisional" to "selected" and unblocks the Phase C
Python-only `--irep2-native-body` prototype.

---

## Appendix B — Phase B result: Q-LOC is 100 % inheritance, D3 is a GO (2026-07-05)

Deliverable (2). Phase A left one empirical question: does the pipeline ever
*consume* a genuinely per-operand-distinct location, i.e. do the multi-line /
macro shapes §2 suspected actually reach `goto_convert` with a distinct operand
location? The answer, measured on the **shipped binary** (no rebuild — see the
mechanism note below for why the corpus percentage is 100 % *by construction*):

**Mechanism (why the rate is 100 %, not a sample).** The body round-trip at
`goto_convert_functions.cpp:184` (`migrate_expr_back(get_value2())`) is
**unconditional** (V.4.4b) and IREP2 value nodes carry no location, so every
value operand reaches `restore_value_locations` (`:187`) **location-less**;
restore then stamps each with its enclosing statement's location (pure
inheritance). So the operand locations `goto_convert` consumes today are
*already* the inherited statement locations — the frontend's distinct operand
locations, if any, were discarded one line earlier. D3 (an IREP2-native pass that
inherits the `code_*2t::location` at consumption) therefore reproduces today's
bytes *by construction*, for any corpus.

**Adversarial confirmation on the suspected counterexamples.** Ran
`--goto-functions-only` on the three shapes §2 named as the likely Q-LOC
violators; in every case the generated instructions carry the **enclosing
statement** line, never the operand's own line:

| Probe | Operand on a distinct line | Generated instruction location | Inherited? |
|---|---|---|---|
| in-expression call `int y = 1 +`↵`h(2);` (stmt line 4, `h(2)` line 5) | line 5 | `DECL`/`FUNCTION_CALL`/`ASSIGN` all **line 4** | ✅ |
| macro `int y =`↵`CALL;` where `CALL`≡`h(9)` (stmt line 5, expansion line 6) | line 6 | `FUNCTION_CALL`/`ASSIGN` all **line 5** | ✅ |
| deep nest `int y = a(`↵`b(3));` (stmt line 5, `b(3)` line 6) | line 6 | inner `b(3)` **and** outer `a(...)` both **line 5** | ✅ |

**Verdict — D3 GO; W1-loc is not a fidelity wall.** The V.4 outcome classified
W1 a `RETAIN_BOUNDARY` because "value-operand source locations cannot live in
IREP2." Phases A+B refute that as a *correctness* barrier: the pipeline never
consumes a distinct operand location — it consumes the enclosing statement
location, which lives on the `code_*2t` kinds already. So literal 100 %
(body-through-`goto_convert` with no `migrate_*` back-hop) **is reachable at
byte-identical fidelity** via D3; no per-node location field (D1) and no
located-operand set (D2) is required. What remains is **not** a fidelity
impossibility but the **engineering cost** the V.4 outcome also named: an
IREP2-native `goto_convert` dispatcher that reimplements `convert_block`'s
destructor-stack / `end_location` unwind, goto target-tracking, and the
side-effect hoisting — behind a per-frontend flag, gated on byte-identical GOTO
across every frontend. That cost is real and unchanged; the spike's contribution
is to move W1-loc from "impossible without location-fidelity loss" to "possible,
priced as a large cross-frontend program."

**Recommended next step (Phase C).** A throw-away Python-only prototype behind
`--irep2-native-body` that routes the Python body to an IREP2-native
`goto_convert` inheriting `code_*2t::location`, gated on byte-identical GOTO over
`regression/python` + the model-`.py` corpus + an `esbmc-cpp` smoke stratum. Its
purpose is to *measure the implementation cost* now that fidelity is settled — it
is a sizing experiment, not a landable PR, exactly as the V.1k rounds were.

---

## Appendix C — Phase C progress (in flight)

Deliverable (3), grown incrementally (the V-track "one `code_*2t` kind at a
time" cadence) rather than as a single throw-away branch. The default-off
`--irep2-native-body` seam and dispatcher `convert_native_rec` land on master
and grow one supported kind per PR; each PR gates on **byte-identical
`--goto-functions-only` flag-on vs flag-off** across a cross-frontend corpus
(the §5 gate 1), with the native path proven reachable (C-Live) by temporary
instrumentation.

| PR | Kinds made native | Notes |
|---|---|---|
| #5866 | `code_block2t` (decl-free), `code_skip2t` | The seam + the decl-free structural leaves; destructor stack never touched. |
| #5896 | `code_assign2t` (side-effect-free, non-atomic, non-code-typed source) | The first **value statement**. Guarded by the exact predicates `convert_assign` uses (`has_sideeffect` on both operands, `rhs.type().is_code()`, `is_atomic_symbol`/`has_atomic_read`); anything richer falls back. Design **D3**: the statement's own `code_assign2t::location` is carried; the side-effect-free reduction means the stored instruction is `code2` itself (value-operand locations are dropped by `migrate_expr` regardless), so no round-trip and no `restore_value_locations` stamping is needed. |
| #5897 | `code_expression2t` (side-effect-free, non-code, non-ternary operand) | The second value statement → one `OTHER`. Guard mirrors `convert_expression`'s non-`OTHER` branches: `has_sideeffect(op)` (lowered), `op.is_code()` (re-dispatched via `convert()`), `op.id()=="if"` (top-level ternary peeled unconditionally into `convert_ifthenelse` at `goto_convert.cpp:507`). Same D3 store-`code2` emission as assign, but located at the statement's own `code_expression2t::location` (which `convert_expression` reads off the operand post-`restore_value_locations`); falls back when that location is nil/empty-file so an inherited block location is never lost. |
| #5899 | `code_decl2t` (trivial type; non-static/non-code/non-array; side-effect-free non-ternary non-`temporary_object` init) | The **first shared-state handler**: the block handler now manages `targets.destructor_stack` exactly as `convert_block` does (save → convert children → `unwind_destructor_stack` at `end_location` → restore), and the decl handler emits `DECL` + optional side-effect-free `ASSIGN` and pushes one scope-exit `code_dead` (the block's unwind turns it into the `DEAD`). DECL/ASSIGN are built from the operand type `migrate_type_back(decl.type)`, the `code_dead` from `s->get_type()` — matching `convert_decl`'s two type sources. Fallback restores the stack on every exit so `goto_convert_rec` re-runs clean. Unlocks real function bodies with locals. The `convert_block` "unreachable → skip destructors" guard is omitted (no native statement emits a trailing unconditional goto — reinstate when `return`/`goto`/`break` land). |
| #5919 | `code_return2t` (value return; side-effect-free non-code non-ternary value) | The **third value statement**, and the **first native kind to emit a trailing unconditional goto**. Mirrors `convert_return`'s plain path: a value-returning function (`targets.has_return_value`) emits a `RETURN` carrying the statement's own `code_return2t` (D3 store-`code2` — `migrate_expr` drops the value-operand location `restore_value_locations` stamped, so it round-trips to `code2`, exactly as assign/expression) followed by an unconditional `GOTO` to `targets.return_target`; falls back on a side-effect value, a `cpp-throw` return value, a top-level ternary, and a missing value in a value-returning function (nondet replacement). The trailing goto forces the block handler to **reinstate `convert_block`'s "unreachable → skip destructors" guard** the decl row deferred, so a local's scope-exit `DEAD` after the goto is suppressed byte-for-byte. A valued tail return (`int f(){ return 42; }`) is a decl-free top-level block, so the native path is exercised despite the valueless-return trap below. |
| #6086 | `code_ifthenelse2t` (side-effect-free guard; both branches convert natively; not a shape `generate_ifthenelse` folds into a guarded assertion) | The **first branching kind**. Reproduces the general (unfolded) branch shape only — `v: if(!c) goto y/z; w: P; [x: goto z; y: Q;] z: ;` — built directly from `cond`/`then_case`/`else_case` (the branch guard is `not2tc(cond)`, exactly `migrate_expr(gen_not(guard))` since the guard is required side-effect-free). Falls back on: a side-effecting guard or any of the branch-coverage instrumentation options (both routes `convert_ifthenelse` handles via `remove_sideeffects`/coverage-preserving guard rewrites this kind does not reproduce); a branch whose destructor-stack size changes across the recursive `convert_native_rec` call (a bare non-block branch leaking an unpopped scope-exit `code_dead`); and — unless `--validate-violation-witness` is set — a branch that reduces to a lone `assert(false)` (or, else-less, a leading one) `generate_ifthenelse` would fold directly into the guard instead of emitting the general shape. Every prior kind emits ≥1 instruction, so the "flip if the then-branch is empty" path never triggers here. Nested `if`s (with or without an `else`, braced or not) compose for free through the existing recursion. |
| #6087 | `code_while2t` (side-effect-free condition; body converts natively with no `break`/`continue`) | The **first looping kind**. Reproduces `convert_while`'s shape exactly — `v: if(!c) goto z; x: P; y: goto v; z: ;` — with `targets.set_break(z)`/`set_continue(y)` called (unused until break/continue support lands, below) so the sequencing doesn't need to change when they land. **Empirical finding**: a plain C assignment *statement* (`x = y;`) is not `code_assign2t` at the statement level — it is a `code_expression2t` wrapping a side-effecting `sideeffect_assign2t` expression, which that kind's own (pre-existing, #5897) guard correctly refuses. So a C loop body built from ordinary reassignments always falls back whole; Python's assignment is a genuine statement (not expression-wrapped), so a Python `while` loop converts natively. C-Live proven on both frontends via a throwaway probe (reverted); the Python case additionally proven byte-identical + verdict-correct end-to-end. |
| #6088 | `code_break2t`, `code_continue2t` | An unconditional `GOTO` to the enclosing loop's break/continue target, preceded by `unwind_destructor_stack` down to `break_stack_size`/`continue_stack_size` — calls the inherited `goto_convertt` method directly (it already leaves `targets.destructor_stack` unchanged by design: the swap-back at the end of the 4-arg overload) rather than reproducing it. **Correction to the #6087 row above**: "every non-`break`/`continue` Python `for` loop converts natively" was wrong — the preprocessor's desugared `for` loop's *condition itself* calls `ESBMC_range_has_next_(...)`, a function call, so `has_sideeffect(cond)` is true and the whole loop falls back regardless of break/continue (confirmed: a `for`-loop with no `break`/`continue` at all also falls back). This kind's practical win is therefore narrower than framed — literal Python `while` loops with `break`/`continue` — until `code_function_call2t` support lands and unblocks `for`. C-Live proven (break, continue, and nested break/continue at different loop levels); byte-identical on 1000 sampled tests. |
| #6101 | `code_function_call2t` (narrow slice: return value unused, plain named callee, side-effect-free arguments) | A bare `foo();` **statement** call — legacy's `code_function_callt` used directly as a statement (not the `sideeffect2t` shape a call takes in *expression* position, e.g. a `while` condition) migrates straight to this kind, distinct from `code_expression2t`. `!result_is_used` means `do_function_call`'s temp-symbol machinery (`return_value$_<fn>$<counter>`, keyed by a shared, order-sensitive counter — the byte-identity risk this track has been avoiding) is never entered, so this slice carries none of that risk. Requires the callee symbol to have a real body (`get_value().has_operands()`) so every `do_function_call_symbol` builtin special-case (`assume`/`assert`/`loop_invariant`/etc. — reached only when the symbol has *no* body) is excluded structurally, matching legacy's own dispatch order; falls back on everything else. **Does NOT unlock `for` loops** — a call used as the loop *condition* (`ESBMC_range_has_next_(...)`) is an expression-position `sideeffect2t`, not this kind, so the while handler's existing `has_sideeffect(cond)` guard still gates it (confirmed empirically: an earlier framing of this row overstated the win before this correction was written). The real gain is call-statement-containing `while`/`if` **bodies** no longer falling back solely because of a bare call inside them. C-Live proven on both frontends; byte-identical on ~1,000 sampled tests plus 5 hand-verified cases (two apparent automated-sweep "mismatches" traced to parallel-execution flakiness in the sweep harness itself — Python's per-run random AST-gen tmpdir path and a pointer-address-keyed `ESBMC_unpack_temp_NNNN` name both leak into `--goto-functions-only` output and must be normalized before diffing, and a rare additional divergence appeared only under 8-way parallel execution, never when the same two tests were re-run sequentially — logged as a sweep-tooling gotcha, not a code defect). |

| #6106 | `code_while2t` (side-effecting condition now also converts), `code_assign2t` (rhs a plain result-used function-call sideeffect), `code_assert2t`, `code_assume2t` | **Correction to the #6088/#6101 rows' framing**: the `for`-loop unlock is *not* an expression-position call inside a `while`/`if` guard — empirically, Python's preprocessor never puts a side-effecting call directly in a loop condition at all. A `for x in range(...)` desugars to `while True: if not ESBMC_range_has_next_(...): break`, and even a direct Python `while <call>():` is rewritten the same way before goto_convert ever sees it (confirmed by dumping the IREP2 body: `code_while2t::cond` is the literal `True`/`1` in every Python case tried). So **`code_while2t`'s side-effecting-condition support here is real but targets a different, narrower case — a C/C++ `while` whose condition is directly a call (`while (has_more())`) — not the Python `for`/`while` desugaring.** The actual `for`-loop blocker turned out to be two statements the desugared preamble emits before the loop: `code_assert2t` (`range()`'s step-nonzero check — a plain kind with zero prior support of any shape) and `code_assign2t` with a `sideeffect2t(function_call)` rhs (`has_next = ESBMC_range_has_next_(...)`, legacy's `convert_assign` special-case at `goto_convert.cpp:998`, dispatching straight to `do_function_call` — a *third*, distinct shape from both `code_function_call2t` and a side-effecting `while` guard). Landing `code_assert2t`/`code_assume2t` (trivial: mirror `convert_assert`/`convert_assume`'s single-instruction emission; `guard` is already `expr2tc` so no round-trip) unblocked the range-step assert; landing the `code_assign2t` extension (delegating to the *existing* `do_function_call` member — not reimplemented — exactly as `code_function_call2t`'s ret-unused slice already established the pattern for) unblocked the `has_next` assignment. **This is what finally converts a real `for x in range(...): ...` loop natively, C-Live-proven and byte-identical.**<br><br>Doing this safely surfaced a genuine architectural gap: `do_function_call`/`remove_sideeffects` allocate temp symbols from the per-function `tmp_symbol` counter and add them to `context` — but a native attempt that calls them and then *later* hits an unsupported statement in the same function discards its `dest` and falls back to `goto_convert_rec`, without undoing those allocations. The abandoned temp's counter increment would then be visible to the *next* temp the legacy fallback creates in that function, breaking byte-identity in exactly the combination this row exercises (`while (has_more())` followed by an ordinary C reassignment statement, still unsupported by `code_expression2t`, forcing a fallback *after* the while-cond's `do_function_call`-adjacent work already ran). Fixed with two new `contextt` primitives — `mark()`/`erase_since()`, O(k) in symbols added rather than an O(n) table scan — and a snapshot/restore of `tmp_symbol.counter` + `context` around `convert_function`'s native/fallback dispatch; empirically confirmed via `while (has_more()) { total = total + 1; ...(unsupported)... }`, where the fallback's `return_value$_has_more$1` numbering is unaffected by the discarded native attempt. |

| #6174 | *(fix)* locate the side-effecting `while` condition | `generate_conditional_branch`/`remove_sideeffects` read each emitted instruction's location off the **operand** being lowered, not off the statement; IREP2 values carry no location, so #6106's back-migrated condition arrived unlocated and `while (t--)` lowered to unlocated `ASSIGN`s. Fixed by `stamp_value_locations(cond_legacy, location)` — the same helper `restore_value_locations` uses on the legacy path. Reachable before #6175 only for a function containing no assignment statements (anything else fell back wholesale), which is why it went unnoticed; #6175 makes it broadly reachable, so it lands first. |
| #6175 | `code_expression2t` wrapping a plain `sideeffect_assign2t` (operator `assign`; side-effect-free non-ternary operands) | The **C/C++ assignment statement**. `x = y;` is not a `code_assign2t` at statement level in C/C++ — the frontends emit an *expression statement* wrapping a `sideeffect_assign2t` (the empirical finding first recorded in the #6087 row), so every ordinary reassignment hit `code_expression2t`'s `has_sideeffect(op)` guard and dragged its whole enclosing function back to `goto_convert_rec`. `convert_expression`'s else-branch hands such an operand to `remove_sideeffects`, whose `"assign"` arm strips the wrapper and calls `convert_assign` on the two operands, then nils the wrapper (`result_is_used` is false) so no `OTHER` follows. This handler reproduces that by **calling the inherited `convert_assign` with the same `code_assignt` legacy builds** — not reimplementing it — so the atomic-dispatch and code-typed-source branches stay byte-identical for free, and the `tmp_symbol`/`context` rollback added in #6106 already covers any temp they allocate before a later statement forces a fallback. Location precedence mirrors `convert_expression`'s round-trip restore: the `sideeffect_assign2t`'s own location, falling back to the enclosing statement's when the round-trip left it unlocated. Compound assignments (`+=`, `-=`, …) are excluded — `remove_assignment` synthesizes a fresh binary rhs (with a bool-operand typecast dance and an RMW-atomic path) that is a separate slice. **This is the kind that first takes a real C function body native end to end**: `int f(int a,int b){int x=a; x=x+b; g=x; while(x<10){x=x+1;} if(x>0){g=x;}else{g=0;} return g;}` converts entirely natively, where before this row it fell back on its first reassignment. A code-typed source is guarded out, matching the `code_assign2t` arm — it would re-enter the legacy dispatcher via `convert_assign`'s `convert(to_code(rhs))`, which can emit labelled gotos `try_convert_body_native` assumes the native subset never produces. C-Live proven. Byte-identity: a 696-test `regression/{esbmc,cbmc}` sweep returns the **same 9 mismatches with and without this patch** — all of one pre-existing cluster (an unlocated DECL-initializer `ASSIGN` printing blank on the legacy path vs `no location` natively, a nil-vs-empty `locationt` distinction unrelated to this kind and fixed in #6176, which takes the sweep to **0/696**). Note the sweep needs three normalizations to be meaningful — timing lines (*both* `creation time` and `processing time`), the per-run hex temp-header dir (`esbmc.NNNN-NNNN-NNNN`), and `ESBMC_unpack_temp_NNNN`; under-normalizing yielded 309 and then 109 false mismatches before the real signal was visible. |

| #6176 | *(fix)* materialise the decl-initializer ASSIGN location | `convert_decl` reads the companion ASSIGN's location through `codet`'s **mutable** `location()` accessor, which materialises an empty (id `""`, non-nil) `#location` when the declaration has none; `code_decl2t::location` stays properly nil, so the native path emitted nil where legacy emitted empty-but-present (the GOTO dump renders these as `no location` vs blank). Reached by a compound literal, whose synthesised variable has no location. The DECL itself keeps the nil location in both paths — `convert_decl` emits it *before* that mutable access happens — so only the ASSIGN needed the change. With this the flag-on/flag-off sweep is **0 mismatches / 696 tests**, clean for the first time on this track. |

| #6177 | `code_expression2t` with **any** side-effecting operand (generalizes the #6175 slice) | Reproduces `convert_expression`'s else-branch verbatim instead of special-casing one shape: stamp the statement location onto the operand, hand it to the inherited `remove_sideeffects` with `result_is_used` false, emit an `OTHER` only if anything survives. Each sub-case therefore stays byte-identical for free — `convert_assign` for `=`, `remove_assignment`'s synthesized rhs for `+=`/`-=`/…, `remove_pre`/`remove_post` for `++`/`--`, `do_function_call` for a discarded call result — and the #6175 special case is deleted rather than extended. The enabling step is `stamp_value_locations(op, expr_stmt.location)`: IREP2 values carry no location and `remove_sideeffects` reads each emitted instruction's location off the operand being lowered, so without it every lowered instruction comes out unlocated. That is the same defect #6174 fixed pointwise for the side-effecting `while` condition; this is its general form, and the two are the only places the native path hands a value expression to a legacy lowering helper. A function mixing compound assignments, pre/post increment and decrement, and a discarded call statement now converts natively end to end. A `temporary_object` anywhere in the operand is guarded out: its scope-exit entries die at the end of the full expression rather than at block exit (C++ [class.temporary]/4, #6075/#6076), and reproducing that destructor-stack interaction natively is a separate slice. This was found empirically — `g = use(T(a));` dropped the temporary's destructor entirely — and is the reason the sweep corpus was **extended to C++** for this row: `regression/{esbmc,cbmc}` is C-only, so it could not have caught it. Sweeps: **0 mismatches / 696** on `regression/{esbmc,cbmc}`, holding the clean baseline #6176 established, and **the same 3 pre-existing mismatches with and without this patch** on 439 `regression/esbmc-cpp/{cpp,cbmc}` tests (a static-initializer ASSIGN in `__ESBMC_main` located natively but blank on the legacy path — present on the base branch, tracked separately). |

| #6178 | `code_goto2t`, `code_label2t` (+ a `targets` rollback fix) | General C `goto`. The goto emits a GOTO carrying `code2` and records `(instruction, destructor_stack)` in `targets.gotos`; the label converts its sub-statement, takes the first emitted instruction as the target, and registers `(target, destructor_stack)` in `targets.labels`. `try_convert_body_native` already ran `finish_gotos`, which resolves the pair and emits the scope-exit `DEAD`s when the jump leaves a deeper stack than the label sits at — so forward, backward, and out-of-block jumps all work without new machinery. A label matching `--error-label` falls back (`convert_label` turns it into an `ASSERT(false)` with property metadata this handler does not reproduce).<br><br>**The rollback fix is the load-bearing part.** `targets.labels`/`gotos`/`cases` hold `goto_programt::targett` iterators *into the native program*. When a later statement forces a fallback that program is discarded, and `finish_gotos` then dereferences dangling iterators once the fallback rebuilds the body — a segfault, not merely stale state. `convert_function` now snapshots and restores the whole `targetst` alongside `tmp_symbol.counter`/`context`. Restoring wholesale rather than clearing the two fields this dispatcher populates is deliberate: `remove_sideeffects` can re-enter the **legacy** `convert()` (a GCC statement expression lowers that way), registering labels and switch cases the native layer never sees and therefore cannot enumerate. That path made the crash reachable from #6177's generalized expression handler with no `goto` in the source at all. Sweeps: **0 mismatches / 959** C, and the same 3 C++ mismatches as the base branch. |

| #6179 | *(hardening)* thread `inherited` through `convert_native_rec`; two cleanups | `restore_value_locations` is **top-down**: it computes `here` from the statement's own location or the enclosing one, and `if (here.get_file().empty()) return;` skips an entire subtree — so a located statement inside an *unlocated* block is stamped by neither it nor anything below it. The native stamp sites consulted only the statement's own location, so they could stamp where the legacy path would not. `convert_native_rec` now threads an `inherited` location and both sites stamp via a shared `effective_location()`, encoding that invariant instead of relying on the corpus to expose a difference. **This fixes no observed divergence** — sweeps are bit-identical to the previous row (0/959 C; same C++ set) — it makes the two paths agree *by construction*, which is what the byte-identity contract needs. Also folds in two review findings: the stamp call moved inside the `has_sideeffect` branch (it was dead work on the far commoner side-effect-free path, which emits from `code2`/`expr_stmt.location` directly), and the `OTHER` node is built as `code_expressiont(op)` rather than re-migrating `code2` only to overwrite its operand.<br><br>**Correction to the #6177 row.** The 3 `esbmc-cpp` mismatches there were attributed by a review pass to #6174's stamping. They are not: they reproduce on **plain master**, before any row in this table. An `ASSIGN` in the synthesized `__ESBMC_main` static-initializer body comes out located natively and blank on the legacy path, and no handler in this dispatcher emits it — the cause is upstream of Phase C entirely and needs its own investigation. |

| [#6231](https://github.com/esbmc/esbmc/pull/6231) | `code_dowhile2t` (side-effect-free condition; body converts natively) | Reproduces `convert_dowhile`'s shape — `w: P; y: if(c) goto w; z: ;` — with the condition restricted to the side-effect-free case, exactly as `code_while2t` was first sliced. That restriction is what collapses the continue target onto the conditional goto: legacy runs `remove_sideeffects` on the condition and makes the *first emitted instruction* the continue target, which with nothing to lower is the branch itself. The condition needs no round-trip (`code_dowhile2t::cond` is already an `expr2tc`, so it goes straight into the instruction guard) — but its *location* does need deriving. `convert_dowhile` reads it from the operand via `code.op0().find_location()`, which on the legacy path is whatever `restore_value_locations` stamped there, i.e. the governing statement location; where that top-down walk skips an unlocated subtree the operand stays bare and `find_location()` reports the **nil** irep, which is not the same thing as a default-constructed empty-id `locationt` (the distinction #6176 turned on), so both cases are reproduced explicitly. The body is converted with the same destructor-stack-size check the `if`/`while` arms use, and an empty body falls back rather than take `instructions.begin()` on an empty program. C-Live proven: `int f(int a,int b){int x=a; do{x=x+b;}while(x<10); g=x; return g;}` converts natively end to end, as does a do/while whose body uses `break` and `continue` and a nested pair. Sweeps: **0 mismatches / 1000** on `regression/{esbmc,cbmc}`, and on 612 `esbmc-cpp/{cpp,cbmc}` only the pre-existing `__ESBMC_main` pair.<br><br>**Correction to the #6177/#6179 rows — the known C++ divergence set is 2, not 3.** Those rows record `cpp_sum_class`, `cpp_sum_class_bug` *and* `ch19_1`. `ch19_1` was never a code divergence: its source prints `__DATE__`/`__TIME__`, so two runs seconds apart embed different string literals — reproduced **flag-off vs flag-off**, which no dispatcher change can cause. It needs a fifth sweep normalization (or exclusion) alongside the three already documented; a fourth apparent mismatch, `github_4397_cstdlib_namespace`, was likewise noise — the per-run temp header dir leaks into *anonymous typedef names* (`__anon_typedef_div_t_at_/…/esbmc_346d-823e-7b35/headers/stdlib_h__13_9`) in an underscore-separated form the documented regex (`esbmc[a-z-]*[.-]…`) does not match. The two genuine ones were re-confirmed by building **plain master** (`def066c765`) and reproducing the identical one-line diff there, rather than by citing this table. |
| [#6235](https://github.com/esbmc/esbmc/pull/6235) | `code_for2t` (side-effect-free, non-nil condition; init, iteration statement and body all convert natively) | Real C/C++ `for` syntax, reproducing `convert_for`'s `A; v: if(!c) goto z; w: P; x: B; y: goto v; z: ;`. As with `code_while2t`/`code_dowhile2t` the condition is restricted to the side-effect-free case, which is what makes the back-edge target `u` collapse onto the guard `v` (legacy points `u` at the first instruction the condition's `remove_sideeffects` emits, and with nothing to lower that program is empty); a condition-less `for(;;)` is excluded because legacy migrates the nil operand straight into the guard.<br><br>Three ordering details in `convert_for` are load-bearing and are reproduced rather than rationalised. (1) The **iteration statement is converted before the body**, and before `set_break`/`set_continue` — both may allocate from the shared `tmp_symbol` counter, and that numbering is observable in the output, so converting them in the natural reading order would break byte identity exactly where the loop mixes calls into both. (2) A **nil iteration statement still emits a `SKIP`**, located at the statement, because that is the instruction the continue target points at — emitting nothing would leave `targets.set_continue` pointing at the following statement. (3) The **init is emitted straight into the caller's `dest`**, outside the break/continue save, and any scope-exit `code_dead` a declaration in it pushes is deliberately left for the enclosing block to unwind — matching legacy, where the C99 `for (int i = ...)` scope is the enclosing block's business. The emission order (v, w, x, y, z) differs from the conversion order, as it does in legacy.<br><br>C-Live proven, and this is the row where ordinary C loop code goes native: `for (int i = 0; i < n; i++)` with a declaration in the init, a nested pair, one using `break`/`continue`, and one with the iteration statement omitted all convert **fully natively** end to end — `i++` works because #6177 generalised the expression statement. Sweeps: **0 mismatches / 1000** on `regression/{esbmc,cbmc}`. Lands after the `code_dowhile2t` row (#6231), independently of it. |

| [#6239](https://github.com/esbmc/esbmc/pull/6239) | `code_switch2t`, `code_switch_case2t` (side-effect-free switch value; body converts natively) | The **last of the structured-CF set**, reproducing `convert_switch`'s `<LOCATION>; if(v==x) goto X; …; goto d; X: Px; …; d: Pd; z: ;`. The switch value is restricted to the side-effect-free case, exactly as every other value-carrying kind here is, so `convert_switch`'s `remove_sideeffects` preamble is empty and each guard reads the value directly; the guard itself is built by calling the inherited `case_guard` on `migrate_expr_back(value)` — the same legacy `exprt` the round-tripped path hands it — rather than synthesising an IREP2 disjunction, so the `or`/`=` shape and the `migrate_expr` back into `instruction::guard` stay byte-identical for free.<br><br>Two `convert_switch` details are reproduced rather than tidied. (1) It clears `targets.cases` but **not** `targets.cases_map`; both are saved and restored by the inherited `break_switch_targetst`, so a nested switch behaves as it does on the legacy path and the asymmetry is preserved instead of corrected. (2) The default jump's location is read off `targets.default_target->location`, which is the trailing `z` SKIP when the switch has no `default` arm — so a default-less switch locates that GOTO at the switch statement, not at any arm.<br><br>The arm handler mirrors `convert_switch_case`: convert the sub-statement, insert a SKIP at the front as the arm's jump target, and register it in `cases`/`cases_map` (consecutive labels on one statement accumulate into one entry) or as the default. Its target SKIP is located at the **sub-statement**, which on the legacy path is `code.code().location()` — read here from the sub-statement's own `code_*2t::location` via a new `statement_location` helper, since `restore_value_locations` stamps value operands and never statement nodes. An arm whose sub-statement emits no instruction falls back: `convert()` appends a SKIP in that case and `convert_native_rec` does not, and the arm's target must be a real instruction (the same guard the `code_label2t` row uses).<br><br>C-Live proven on consecutive case labels, fallthrough, a default-less switch, a default that is not last, a nested pair, a switch inside a `for` mixing `break` and `continue`, an arm introducing a scoped declaration, and an empty body — all convert **fully natively**. Sweeps: **0 mismatches / 1001** on `regression/{esbmc,cbmc}`. |

| *(this branch)* | *side-effecting conditions for* `code_for2t`, `code_dowhile2t`, `code_switch2t` | Each of those kinds was first sliced to the **side-effect-free** condition, the shape where `remove_sideeffects` emits nothing and the loop's back-edge / continue target collapses onto the guard instruction. This lifts that restriction, so the preamble is real and the target has to point at **its** first instruction instead. `code_while2t` needed no change — #6106 already handled its side-effecting condition, and #6174 located it — but the other three use a different legacy mechanism: `convert_while` routes through `generate_conditional_branch` (which decomposes `&&`/`||` and lowers at the leaf), whereas `convert_for`/`convert_dowhile`/`convert_switch` call `remove_sideeffects` on the condition directly and migrate the **lowered** expression into the guard. Reproduced as written, one branch per kind, sharing the `migrate_expr_back` → `stamp_value_locations` → `remove_sideeffects` → `migrate_expr` sequence the while row established.<br><br>Two things are load-bearing. (1) **Where** the lowering runs: legacy does it *before* saving break/continue and — in `convert_for` — before converting the iteration statement, and it can allocate from the shared `tmp_symbol` counter, whose numbering is observable, so the order is kept rather than moved to where it reads better. (2) **Which target moves**: `for` points `y: goto u` at the preamble's first instruction (falling back to the guard `v` when empty), `do`/`while` makes that instruction the continue target (falling back to the conditional goto `y`), and `switch` simply prepends its preamble ahead of the case-guard chain — no target points into it, which is why the switch case is the cheap one. The side-effect-free paths keep their existing emission (guard straight from the `expr2tc`, no round-trip) so the byte-identity already proven for them is not perturbed.<br><br>C-Live proven on `for (; has_more(); )`, `for (int i = 0; has_more(); i++)`, `do … while (t--)`, `do … while (has_more())`, `switch (has_more())` and `switch (a++)` — every one converts **fully natively** (`try_convert_body_native` returns true), not merely reaching the handler. Sweep: **0 mismatches / 1101** on `regression/{esbmc,cbmc}`. |

*Next:* the residual `code_return2t` **trap** (a missing value in a
`has_return_value` function, i.e. a `return <expr>;` whose `<expr>` evaporates
under `remove_sideeffects` because it calls an undefined function — not a bare
`return;`, which is either a frontend-elided no-op in a void function or a hard
C parse error in a non-void one) remains unhandled and low-priority — it is
unclear whether it is reachable at all outside the undefined-function-call
shape, and constructing a minimal repro needs more investigation before it is
worth another native-kind slot. With `code_switch2t` above the **structured-CF
set is complete** — block, decl, assign, expression, return, if/else, while,
do/while, for, switch, break, continue, goto, label, assert, assume, call all
convert natively, and with the side-effecting-condition row above every one of
them accepts a side-effecting condition too. What remains is a
`temporary_object` in an expression statement (its scope-exit entries die at
end-of-full-expression rather than at block exit, #6177), the C++-only kinds —
`code_cpp_catch2t` / `code_cpp_throw2t` and the `try`/`catch` target machinery —
and the `code_asm2t` / `code_printf2t` leaves.
