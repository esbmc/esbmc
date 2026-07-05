# Spike — IREP2-native value-operand location carriage (the W1-loc keystone)

**Program:** repo-wide "IREP2-native frontend→goto pipeline" (the Part V umbrella, #4715).
**This issue:** the mandatory first spike. Read-only investigation + one throw-away prototype.
**Owner:** TBD. **Status:** Phase A executed (2026-07-05) — census complete, **D3
provisionally selected, D1 ruled out**; Phase B corpus measurement pending. See
Appendix A. **Refs:** #4715, Part V of `docs/irep2-migration.md`.

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
