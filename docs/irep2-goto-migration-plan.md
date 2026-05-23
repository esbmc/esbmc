---
title: "Migration Plan: Legacy IREP → IREP2 in the goto-program pipeline"
status: draft
date: 2026-05-22
owner: goto-programs
---

# Migrating the goto-program pipeline from legacy `irept` to IREP2

## 0. TL;DR for reviewers

The premise "goto-programs still depend on legacy IREP in several classes" is
correct, but the dependency is **narrower and more structured than a rewrite**.
The *core* instruction representation is already IREP2:

- `goto_programt::instructiont::code` is `expr2tc` (`src/goto-programs/goto_program.h:93`)
- `goto_programt::instructiont::guard` is `expr2tc` (`goto_program.h:105`)
- loop invariants / assigns targets are `std::list<expr2tc>` (`goto_program.h:108,111`)

Legacy `irept`/`exprt`/`typet`/`codet` survives only at **six boundaries**.
The migration is therefore "drain each boundary, in dependency order, behind an
adaptor, with differential validation" — not a big-bang core rewrite. This is
inherently friendlier to incremental review and rollback, which is exactly what
the constraints ask for.

The single most important sequencing fact: **do not touch serialization or the
symbol table until everything that consumes them is already IREP2-clean**. They
are the foundation; they are also the highest-risk subsystems (hashing,
structural sharing, on-disk format compatibility).

---

## 1. Dependency and impact analysis

### 1.1 The six surviving legacy boundaries

| # | Boundary | Where | Legacy type | Why it persists | Risk |
|---|----------|-------|-------------|-----------------|------|
| B1 | **Frontend → goto lowering input** | `goto_convert*` consume `codet` (`goto_convert_class.h:18,37,131-167`; `goto_convert.cpp:117`) | `codet`/`exprt` | The clang/python/etc. frontends emit `symbolt::value` as `codet`; `goto_convert` migrates per-instruction (`goto_convert.cpp:136` `migrate_expr`) | Medium |
| B2 | **Symbol table** | `symbolt::value` (`expr2tc`), `symbolt::type` (`type2tc`) (`src/util/symbol.h`) | IREP2 | **Done.** Storage flipped to IREP2 source-of-truth with lazy legacy caches (S5a #4735 → V1 #4737 → V2 #4741 → S6). End-state design in `docs/irep2-symbol-table-s6-plan.md`. | (closed) |
| B3 | **`goto_functiont::type`** | `code_typet type` (`src/goto-programs/goto_functions.h:24`) | `code_typet` (irept) | Function signature still stored stringy | Medium |
| B4 | **Goto-binary serialization** | `goto_program_irep.cpp:4-57`, `goto_program_serialization.cpp`, `goto_function_serialization.cpp`, `read_bin_goto_object.cpp`, `write_goto_binary.cpp`, `src/util/irep_serialization.{h,cpp}` | `irept` | On-disk format is `irept`; converted via `migrate_expr_back`/`migrate_expr` on each I/O | **High (format compat + hashing)** |
| B5 | **`rw_set` data-race pass** | `rw_set.h:17-18` stores `exprt original_expr`, `exprt guard`; `rw_set.cpp:8-72` operates in `exprt` | `exprt` | Pass never migrated; still computes over legacy expressions | Medium |
| B6 | **Analysis passes that build `exprt` then migrate** | `goto_check.cpp:116-565`, `add_race_assertions.cpp:19-238`, `contracts/contracts.cpp` (round-trips `migrate_type_back` at `:1730,1896,2145,2331,2369,2501,2540`), `goto_atomicity_check.cpp:42`, `goto_sideeffects.cpp:13-104`, `destructor.cpp:5-34`, `format_strings.cpp:179`, `assign_params_as_non_det.cpp` | `exprt`/`typet` (transient) | These *produce* IREP2 (they `migrate_expr` before insertion) but *build* in legacy IREP and round-trip through `symbolt` | Medium |

Note B6 round-trips (`expr2tc → exprt → expr2tc`) are *forced* by B2: when a pass
creates a new symbol it must store `exprt`/`typet` into the symbol table, so it
back-migrates. **B6 cannot be fully eliminated until B2 is done.** This is the
core ordering constraint.

### 1.2 Already-migrated (do not touch except to delete adaptors later)

- Instruction `code`/`guard`, loop contracts — `expr2tc` (`goto_program.h`).
- Loop analysis: `loopst.h:14`, `goto_loops.h:14` use `std::unordered_set<expr2tc, irep2_hash>`.
- k-induction: `goto_k_induction.h` uses `guard2tc`.
- Abstract interpretation: `ai.h`, `interval_domain.*`, `gcse.*`, `wrapped_interval.h` operate over `expr2tc`.

### 1.3 The conversion layer (the seam we lean on)

`src/util/migrate.{h,cpp}`: `migrate_expr`, `migrate_type`, `migrate_expr_back`,
`migrate_type_back`. Every boundary above already routes through these. **The
migration strategy is to push these calls outward toward B2/B4 and shrink the
legacy region they bound, rather than to invent new machinery.** When B2 and B4
are migrated, large parts of `migrate.cpp` become dead and can be deleted last.

---

## 2. High-risk areas (call out explicitly)

These are the properties that, if broken, produce *silent unsoundness or
nondeterminism* — the worst failure mode for a verifier. Each gets dedicated
validation in §5.

1. **Serialization / on-disk goto-binary format (B4).** Changing how
   instructions serialize can silently change byte layout, breaking cached
   goto-binaries and cross-version reads. The `irep_serialization` string-pool
   and `reference_convert` sharing logic (`goto_program_serialization.cpp:17`)
   are subtle. Treat the binary format as a *frozen external contract* until a
   deliberate, versioned format bump.
2. **Hashing & structural sharing.** Legacy `irept` and `irep2` have *different*
   hash functions and sharing semantics (`irep2_hash` vs irept's hashing). Sets
   keyed on expressions (`loop_varst`, AI domains) must not change iteration
   order — order changes propagate into SMT formula construction and can flip
   results or timings nondeterministically.
3. **Equality semantics.** `irept` equality is structural-string; `expr2tc`
   equality is typed and considers the type node. Code that compared `exprt`
   for identity may get *different* answers after migration (e.g. two ints of
   different bit-width that stringified equal). Audit every `==`/`operator<`.
4. **Symbol handling (B2).** `symbolt::value` doubles as "code body" and "const
   value"; the Python frontend already has a known dual-role hazard
   (`gotcha_python_symbol_value_static_lifetime`). Migrating B2 must preserve
   the static-initializer vs function-body distinction exactly.
5. **Expression canonicalization.** `simplify`/typecheck behaviour differs
   between the two reps. Migrating a pass must not silently insert/skip a
   simplification step that changes the VC set.
6. **Goto-program transformations producing different instruction sequences.**
   Any pass (B5, B6) whose output instruction list changes — even reordered —
   changes symex paths. Differential goto-binary diffing is the guard here (§5).
7. **Ownership/lifetime.** `expr2tc` is reference-counted copy-on-write; `exprt`
   is value-copied. A pass that mutated an `exprt` in place and relied on the
   copy being independent may, after migration, mutate shared state via COW
   aliasing. Audit in-place mutation sites.

---

## 3. Assumptions in the current code that may break under IREP2

- **In-place mutation independence** (lifetime risk #7). `exprt` subexpression
  edits are local; `expr2tc` edits trigger COW — patterns that grab `exprt &`
  and mutate operands (e.g. `goto_sideeffects.cpp`, `add_race_assertions.cpp`)
  need `expr2tc` rewrite via the visitor / `Forall_operands` analogue.
- **String-keyed lookups.** Code using `.id()`/`.get("…")` string tags assumes
  the stringy schema. IREP2 nodes are typed enums — every `irep_idt` tag access
  must map to an `expr2t::expr_ids` case.
- **`code_typet` introspection** (B3). Passes reading argument types off
  `goto_functiont::type` (`goto_convert_functions.cpp:288,301`,
  `assign_params_as_non_det.cpp:57,115-120`,
  `destructor.cpp`) assume legacy `code_typet`. IREP2's `code_type2t` exposes a
  different accessor surface.
- **Serialization round-trip identity.** Code may assume "write then read"
  yields a byte-identical / pointer-shared graph. IREP2 deserialization rebuilds
  fresh nodes; any reliance on post-read sharing breaks.
- **Symbol value emptiness sentinels.** `symbolt::value.is_nil()` vs an empty
  `expr2tc()` — the nil/none distinction differs and is load-bearing in
  `instructiont::clear` (`goto_program.h:167-168`).

---

## 4. Phased migration roadmap

Each phase is independently shippable, independently revertible, and gated on
the validation in §5 passing **with dual-solver agreement (Bitwuzla + Z3)** and
zero goto-binary diff on the corpus.

### Phase 0 — Instrumentation & safety net (no behaviour change)
- Build the **differential harness** (§5.1): a script that runs ESBMC on a fixed
  corpus, emits goto-binary + counterexample digests, and diffs against a golden
  baseline captured from `master` HEAD *before* any change.
- Add a `migrate_*` **call-site census** as a checked-in report so each later
  phase can show the legacy region shrinking (a measurable checkpoint, §6).
- Capture baselines: full regression corpus, SV-COMP subset digests, goto-binary
  hashes.
- **Exit criterion:** harness reproduces identical digests on two clean builds.

### Phase 1 — Drain pure-analysis legacy (B6 leaf passes, no symbol creation)
Target passes that *build* `exprt` transiently but neither persist symbols nor
touch serialization. Lowest risk, no B2 dependency.
- `format_strings.cpp` (`:179`), `goto_sideeffects.cpp` (`:13-104`),
  `destructor.cpp` analysis (`:5-34`).
- Rewrite to construct `expr2tc` directly (drop the build-then-`migrate_expr`).
- **Exit:** zero goto-binary diff on corpus; these files contain no
  `exprt`/`typet` locals.

### Phase 2 — Migrate `rw_set` (B5)
- `rw_set` is self-contained (race detection). Rewrite `rw_sett::compute/assign`
  to take `const expr2tc &` and store `expr2tc original_expr/guard`
  (`rw_set.h:17-18`). Update its sole caller `add_race_assertions.cpp`.
- This removes one whole legacy island without symbol-table coupling.
- **Exit:** `--data-races-check` corpus shows zero goto-binary diff;
  `rw_set.{h,cpp}` free of `exprt`.

### Phase 3 — Migrate `goto_functiont::type` (B3)
- Introduce IREP2 `code_type2t` storage on `goto_functiont` *behind an accessor*
  that still exposes a legacy view (adaptor) so callers compile unchanged.
- Migrate readers one at a time (`goto_convert_functions.cpp:288,301`,
  `assign_params_as_non_det.cpp`, `destructor.cpp`, contracts) to the IREP2
  accessor.
- Remove the legacy adaptor once no caller uses it.
- **Exit:** `goto_functiont` stores only IREP2 type; corpus diff clean.

### Phase 4 — Symbol table (B2) — the foundational, highest-risk step
This is the linchpin (it unblocks the rest of B6). Do it *most carefully*.
- 4a. Add `expr2tc value2`/`type2tc type2` *alongside* the legacy fields, kept in
  sync by the existing `migrate_*` calls (shadow fields). Assert equality of the
  two views on every access in debug builds (differential self-check).
- 4b. Switch *readers* in the goto pipeline to the IREP2 fields, leaving
  frontends writing legacy (still synced).
- 4c. Switch *writers* (the B6 contract/atomicity passes that currently
  `migrate_type_back`) to write IREP2 directly, dropping the round-trips at
  `contracts.cpp:1730,1896,2145,2331,2369,2501,2540` and `goto_atomicity_check.cpp:42`.
- 4d. Migrate `symbolt::to_irep/from_irep` (`symbol.h:42-43`) — **this couples to
  B4; sequence 4d with Phase 5.**
- **Exit per sub-step:** shadow-field equality assertions never fire across the
  full corpus; zero goto-binary diff.

### Phase 5 — Serialization / goto-binary format (B4)
Do **last**, deliberately, and behind an explicit format version.
- 5a. Keep reading the legacy `irept` format (back-compat) but add an IREP2-native
  writer behind a version byte bump in the goto-binary header.
- 5b. Differential test: write legacy + IREP2 formats, read both, assert digest
  equality of resulting goto-functions.
- 5c. Flip the default writer to IREP2 only after a full corpus round-trips
  identically. Retain the legacy reader indefinitely for old binaries (or gate
  removal on a separate, later deprecation decision).
- **Exit:** new binaries IREP2-native; old binaries still load; corpus digests
  identical across the format boundary.

### Phase 6 — Frontend lowering input (B1) — optional / longest-horizon
- `goto_convert` consuming `codet` is the deepest boundary (touches every
  frontend). Migrate only after Phases 1–5 prove the approach. May be deferred
  indefinitely — `goto_convert.cpp:136` migrating once per instruction is cheap
  and correct.
- **Decision gate:** evaluate whether B1 removal is worth the cross-frontend
  churn, or whether the `migrate_expr` seam at goto-convert is the permanent,
  acceptable boundary.

### Phase 7 — Cleanup
- Delete now-dead `migrate_*_back` paths, adaptors, shadow fields, legacy
  accessors. Only after correctness is proven (§ workflow rule: "cleanup after
  correctness").

---

## 5. Validation & testing strategy

### 5.1 Differential goto-binary diffing (primary guard)
- For every corpus input, dump the goto program (`--goto-functions-only`) and the
  goto-binary; canonicalize and hash. **Any diff between baseline and patched is
  a hard failure** unless explicitly justified (the "deviation must be justified"
  constraint). This catches transformation-order and canonicalization drift
  (risks #5, #6) that pass/fail verdicts alone would miss.

### 5.2 Differential verdict + counterexample testing
- Run the full `regression/` suite (cap 5 min per CLAUDE.md; narrow per phase).
- Compare not just SAT/UNSAT verdict but **counterexample trace digests** —
  a changed trace at an unchanged verdict signals a semantic shift.
- **Dual-solver:** every phase gate requires Bitwuzla *and* Z3 agreement (matches
  the repo's Mode-C dual-solver rule).

### 5.3 Shadow-field equality assertions (Phase 4)
- Debug-build assertions that the legacy and IREP2 views of every symbol are
  semantically equal at each access. Run across the corpus; zero firings is the
  gate.

### 5.4 Serialization round-trip (Phase 5)
- Property test: `read(write(g)) ≡ g` for both formats, and
  `read_legacy(write_legacy(g)) ≡ read_irep2(write_irep2(g))`.
- Cross-version: old binaries from a tagged release must still load.

### 5.5 Hashing/ordering invariants (risk #2, #3)
- Add a targeted test that iterates the expr-keyed sets (`loop_varst`, AI worklists)
  and asserts deterministic order across two runs and across reps.

### 5.6 ESBMC self-verification & sanitizers
- Per repo workflow: `esbmc-verifier` agent on touched C/C++; ASan/UBSan build
  of the unit + regression harness for any pointer/lifetime-touching phase
  (risk #7).
- Unit tests under `unit/` (Catch2) for `migrate.*` invariants.

---

## 6. Measurable checkpoints / milestones

| Milestone | Metric | Gate |
|-----------|--------|------|
| M0 baseline | Golden digests captured; harness reproducible | 2 clean builds identical |
| M1 (Phase 1) | `exprt` locals in 3 analysis files → 0 | Corpus diff = 0 |
| M2 (Phase 2) | `rw_set` `exprt` count → 0 | Race corpus diff = 0 |
| M3 (Phase 3) | `goto_functiont` legacy type readers → 0 | Corpus diff = 0 |
| M4 (Phase 4) | Symbol shadow-field assertions firings | 0 firings, corpus diff = 0 |
| M5 (Phase 5) | Format round-trip + old-binary load | Digests identical; back-compat read OK |
| M6 (Phase 6, opt) | `migrate_expr` calls in goto_convert | Decision recorded |
| M7 (cleanup) | Dead `migrate_*_back` LOC removed | Build green, full regression green |

The **`migrate_*` call-site census** (Phase 0) is the running scoreboard: each
phase must show the count strictly decreasing in its target region and *not*
increasing anywhere else.

---

## 7. Rollback & compatibility considerations

- **Per-phase revert points.** Each phase is a small, self-contained PR set; a
  failed gate reverts that PR without disturbing earlier phases. No phase is
  merged until its gate is green — preserving the no-broken-commits rule.
- **Adaptor layers as rollback insurance.** Phases 3 and 4 keep legacy views
  alive behind adaptors/shadow fields, so a regression discovered downstream can
  be diagnosed by toggling reads back to the legacy view without a full revert.
- **Serialization back-compat is permanent for reads** (Phase 5a): the legacy
  `irept` goto-binary reader is retained even after the writer flips, so existing
  cached binaries and CI artifacts keep loading. Removal of the legacy reader is
  a *separate, later, explicitly-approved* deprecation — not part of this
  migration.
- **Format version byte** gates Phase 5; an IREP2-written binary is unambiguously
  distinguishable from a legacy one, so a misread fails loudly instead of
  silently mis-deserializing.
- **Never squash** (repo rule): each incremental step stays in history for
  bisection if a regression surfaces months later.

---

## 8. Recommended implementation order (summary)

```
Phase 0  Instrumentation + baselines        (no behaviour change)
Phase 1  Pure-analysis leaf passes (B6')     ── lowest risk, no coupling
Phase 2  rw_set / data-race (B5)             ── self-contained island
Phase 3  goto_functiont::type (B3)           ── behind adaptor
Phase 4  Symbol table (B2)                   ── DONE (executed via S0..V2+S6)
           ├ S0 encapsulate accessors           #4724
           ├ S1 chokepoints + cross-check       #4725, #4727
           ├ S3 IREP2 writer chokepoint         #4728
           ├ S4a audit writes / remove ref-gets #4729
           ├ S4b IREP2 lazy cache               #4730
           ├ S2 closeout                        #4731
           ├ S5a type source-of-truth flip      #4735 (+ #4739 hotfix)
           ├ V1 close back-migration coverage   #4737
           ├ V2 value source-of-truth flip      #4741
           └ S6 end-state: lazy caches retained #4742, docs/irep2-symbol-table-s6-plan.md
Phase 5  Goto-binary serialization (B4)      ── versioned, back-compat reader kept
Phase 6  Frontend lowering input (B1)        ── OPTIONAL / deferrable
Phase 7  Cleanup: delete dead migrate paths  ── only after correctness proven
```

Rationale: passes with no symbol-table or serialization coupling go first (cheap
confidence). The symbol table (B2) is the foundation that unblocks the forced B6
round-trips, so it precedes serialization. Serialization (B4) is last because it
is the highest-risk and the only one with an external (on-disk) compatibility
contract. B1 is genuinely optional — the `migrate_expr` seam at goto-convert may
be the right permanent boundary.
