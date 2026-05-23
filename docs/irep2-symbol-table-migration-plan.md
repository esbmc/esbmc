---
title: "B2 execution plan — migrating symbolt::type/value to IREP2"
status: complete
date: 2026-05-22
supersedes-detail-in: docs/irep2-goto-migration-phase4-symboltable.md
tracking-issue: esbmc/esbmc#4715
historical-record-of: "the plan as drafted; the actual execution followed
  the staged path captured in irep2-symbol-table-phase5-plan.md (S5a),
  irep2-symbol-table-vtrack-plan.md (V1, V2) and
  irep2-symbol-table-s6-plan.md (S6 end state)."
---

# Symbol-table (B2) migration — concrete execution plan

This is the code-grounded execution plan for the last and largest boundary of
the goto-program IREP2 migration: `symbolt::type` (`typet`) and `symbolt::value`
(`exprt`) in `src/util/symbol.h:15-16`. It supersedes the *strategy* in
`docs/irep2-goto-migration-phase4-symboltable.md` (which predates the merged
groundwork) with verified facts and a sharper plan.

## What is already true (merged groundwork)

- **B3 / B5 done.** `goto_functiont::type` is IREP2 (#4721); `rw_set` is IREP2
  (#4718/#4719). The remaining legacy in the goto pipeline is overwhelmingly
  `migrate_type(symbol.type)` / `migrate_expr(symbol.value)` — i.e. B2-bound.
- **migrate idempotence is proven** for symbol-relevant kinds (#4722,
  `unit/util/migrate.test.cpp`): `migrate_type(migrate_type_back(T2)) == T2` and
  the expr analogue, with two documented exceptions — `migrate_expr_back` does
  not reconstruct a whole `code_block2t` (a function body migrates *forward*
  only), and `constructor`/`destructor` return types migrate to `empty`. This
  is the lossless property the whole plan leans on.
- **The differential harness exists** (`scripts/irep2-migration/`): every slice
  is gated on zero canonical goto-program diff over a corpus.

## Verified impact analysis

### Access surface (files touching `symbol.value` / `symbol.type`)

| Tree | files | role |
|------|------:|------|
| `util/` | 12 | symbol def, serialization, namespace, type utils |
| `goto-programs/` | 15 | goto-convert, contracts, checks |
| `goto-symex/` | 16 | symex reads signatures/types |
| `clang-c-frontend/` | 4 | writes C symbols |
| `clang-cpp-frontend/` | 5 | writes C++ symbols |
| `python-frontend/` | 26 | **heaviest writer** |
| `solidity-frontend/` | 11 | heavy writer (function bodies, initialisers) |
| `jimple-frontend/` | 5 | writes Java/Kotlin symbols |

≈43 files in the verification **pipeline** (read-dominant), ≈51 across the
**frontends** (write-dominant). This split drives the slicing: pipeline first
(readers), frontends last (writers).

### Four load-bearing facts (each verified this session)

1. **Serialization is opaque.** `symbolt::to_irep` does `dest.type() = type;
   dest.symvalue(value);` and `from_irep` casts straight back
   (`src/util/symbol.cpp` ~95-160). No code introspects the irept structure of
   `value`/`type` during (de)serialization; `reference_convert` dedups purely
   structurally. ⇒ the goto-binary format is **derivable from IREP2 via
   `migrate_*_back`** with no format change, and B2 has **no B4 coupling**.
2. **Writes cannot be intercepted today.** `contextt::find_symbol` returns a
   mutable `symbolt*` (`src/util/context.h:88,92`) and callers mutate
   `sym->value = …` directly. ⇒ a setter chokepoint requires **encapsulation
   first** — this is the hard ordering constraint.
3. **`value`'s role is keyed off `type`, not `value`.** A symbol's `value` is a
   function body iff `type.is_code()` (`goto_convert_functions.cpp:107,113`),
   else a constant initialiser (`python_converter.cpp` static-init:
   `static_lifetime && !value.is_nil() && !type.is_code()`). Nothing checks
   `value.type() == type`. ⇒ preserve the `is_code` discriminator exactly
   (`is_code_type2t` under IREP2); this is the known static-lifetime hazard.
4. **`migrate_*` is lossless on symbols** (proven, #4722) — but only the
   *forward→back→forward* idempotence is guaranteed. Code bodies (`code_block`)
   are forward-only; treat the legacy body as authoritative until symex consumes
   it (it already does, via the goto program).

## Strategy decision

Three candidate end-state models were considered:

| Model | Source of truth | Sync mechanism | Verdict |
|-------|-----------------|----------------|---------|
| **A. Derived read-accessor only** | legacy | recompute IREP2 on each read | Safe but never drops the legacy field; recompute cost in symex hot paths. Insufficient — doesn't reach the goal. |
| **B. Eager shadow fields** (original doc) | legacy | sync `value2`/`type2` on every write | Correct but maintains two fields in lockstep forever during the transition; most moving parts. |
| **C. Cached-derive → native flip** ✅ | legacy → then IREP2 | one lazily-computed IREP2 cache, invalidated by the setter; flip to native at the end | Single source at all times, no dual-write sync, reaches native storage and drops the legacy field. |

**Chosen: Model C.** Because serialization is opaque (fact 1) and migrate is
proven idempotent (fact 4), IREP2 can become the *sole* stored representation
with legacy derived only at the two `to_irep`/`from_irep` points. The transition
uses a **lazily-computed, setter-invalidated IREP2 cache** rather than eager
dual-field sync (model B), which removes the per-write sync burden while keeping
exactly one source of truth at every step. A debug-only cross-check
(`migrate_type_back(cache) == legacy`) runs during the transition to prove
losslessness on *real* program symbols (the #4722 tests only cover synthetic
ones).

## Phased slices

> **All slices below were executed** between #4724 and #4741.
> The actual storage flip and ABI decisions are recorded in
> `irep2-symbol-table-phase5-plan.md` (S5a — type flip),
> `irep2-symbol-table-vtrack-plan.md` (V1, V2 — value-side flip), and
> `irep2-symbol-table-s6-plan.md` (S6 — end-state ABI).

Every slice: builds, full-corpus **zero goto-diff**, dual-solver
(Bitwuzla+Z3) verdict agreement, per-tree ctest, `scripts/check_python_tests.sh`
when Python is touched. Each is an independently revertible PR.

### S0 — Encapsulate accessors (mechanical, no behaviour change)
Add `symbolt::type()`/`value()` const getters and `set_type()/set_value()`
setters; make the fields private (`type_`/`value_`). Migrate all ≈94 call sites
to the accessors. **Split by tree to stay reviewable**, each sub-PR self-contained:
- **S0a** `util/` + `goto-programs/` + `goto-symex/` + `pointer-analysis/`
- **S0b** `clang-c-frontend/` + `clang-cpp-frontend/`
- **S0c** `python-frontend/` (largest)
- **S0d** `solidity-frontend/` + `jimple-frontend/` + remaining (`esbmc/`, `unit/`)

Getters return `const typet&`/`const exprt&` (unchanged type); setters are plain
stores for now. The diff is uniform and mechanical → easy review. Gate: zero
goto-diff per sub-PR. Rollback: revert the sub-PR; no cross-slice coupling.

### S1 — IREP2 cache + read accessors
In `symbolt`: add `mutable type2tc type2_cache_`/`expr2tc value2_cache_`,
computed lazily by `type2()`/`value2()` via `migrate_*` and invalidated in
`set_type`/`set_value`. Add the debug cross-check. **No reader migrated yet.**
Gate: cross-check never fires across the full corpus + SV-COMP subset (the
central evidence that migrate is lossless on real symbols); zero goto-diff.

### S2 — Switch pipeline readers to IREP2
Migrate the ≈20 `migrate_type(symbol.type)` / `migrate_expr(symbol.value)` read
sites in `goto-programs`/`goto-symex`/`util` to `symbol.type2()`/`value2()`.
Preserve the `is_code` discriminator as `is_code_type2t(symbol.type2())`. Perf-
neutral (same migrate, now cached). Gate: zero goto-diff; cross-check silent.

### S3 — Switch writers to IREP2; kill the B6 round-trips
Frontends and instrumentation passes that build IREP2 then `migrate_*_back` to
store legacy (`contracts.cpp` ~8 sites, `assign_params_as_non_det`, etc.) call
`set_type(type2tc)` / `set_value(expr2tc)` overloads that store the cache
directly and derive legacy. Migrate frontend writers cluster-by-cluster
(jimple → clang-cpp → clang-c → python → solidity, smallest-first). Gate:
zero goto-diff per cluster; `check_python_tests.sh`; dual-solver.

### S4 — Flip source of truth + serialization seam
Make IREP2 the stored field; `to_irep`/`from_irep` derive legacy via
`migrate_*_back` / `migrate_*` (no on-disk format change — fact 1). Old
goto-binaries still load. Gate: full corpus diff = 0; round-trip read/write of
both old and new binaries identical.

### S5 — Cleanup
Originally framed as "remove the legacy field" — superseded by the
end-state decision in `irep2-symbol-table-s6-plan.md`: the legacy
`typet` / `exprt` fields are retained as **permanent on-demand caches**.
IREP2 is the source of truth on `symbolt`; the legacy caches are lazy,
zero-cost-on-unused-paths, and source-API-stable for the 640+ accessor
call sites. The cross-check is retained inside `migrate_symbol_*` for
ongoing losslessness evidence.

## Risk by area

- **Serialization (S4)** — highest. Mitigation: fact 1 (opaque) means no format
  change; gate on old-binary load + byte round-trip. Keep the legacy reader.
- **Dual-role / static-lifetime (S2/S3)** — the `type.is_code()` discriminator.
  Mitigation: lift the known hazard to a regression (function symbol + same-named
  `static_lifetime` global + local) before S2; assert `is_code_type2t` parity.
- **Hashing / iteration order** — symbol containers are keyed by `irep_idt`
  (order-independent of value/type), so unaffected; but expr-keyed sets fed from
  symbol values must stay deterministic. Mitigation: harness goto-diff catches
  any reordering (it already caught the Python astgen-path non-determinism).
- **`migrate` non-idempotence corners** (code_block, ctor/dtor return) — fact 4.
  Mitigation: never back-migrate a body block; the cross-check (S1) surfaces any
  real-symbol case the #4722 unit tests missed.
- **COW aliasing** — `expr2tc`/`type2tc` are reference-counted COW; legacy
  `exprt`/`typet` are value-copied. Audit any code that mutated a symbol's
  `value`/`type` sub-tree in place (the setters make these explicit).
- **Perf** — the lazy cache is computed once per symbol and reused; S2 readers
  are perf-neutral vs today's per-read `migrate_type`. Measure symex on a heavy
  benchmark before/after S2.

## Validation strategy

1. **Differential goto-binary diff** (primary) over a corpus spanning C, C++,
   Python, CUDA, contracts — zero diff per slice, with the documented
   non-determinism filters (`__file__`, astgen paths).
2. **S1 cross-check** — debug assertion `migrate_*_back(cache) == legacy` on
   every symbol access; zero firings is the lossless-on-real-symbols proof.
3. **Dual-solver** Bitwuzla + Z3 verdict + counterexample-digest agreement.
4. **Dual-role regression** (new) pinning the static-lifetime / `is_code`
   discriminator.
5. **`scripts/check_python_tests.sh`** for every Python-touching slice.
6. **Extend `unit/util/migrate.test.cpp`** with any real-symbol shape the
   cross-check flags.

## Measurable checkpoints

| Milestone | Metric | Gate |
|-----------|--------|------|
| S0a–d | accessor migration per tree; field private | zero goto-diff; build green |
| S1 | cross-check firings across corpus | **0** |
| S2 | `migrate_type(symbol.type)` reads in pipeline | → 0 (replaced by `type2()`) |
| S3 | `migrate_*_back` in `goto-programs` (esp. contracts) | strictly ↓ |
| S4 | old + new goto-binary round-trip | byte-identical; old loads |
| S5 | legacy `symbolt` field + dead shims | removed; full regression green |

The `scripts/irep2-migration/migrate_census.sh` scoreboard tracks S2/S3
quantitatively (the legacy region must shrink monotonically, increase nowhere).

## Rollback & compatibility

- Per-slice revert points; S0 sub-PRs are mutually independent (no stacking —
  the squash-merge strategy breaks stacks, as #4719 showed).
- Through S3 the legacy field stays authoritative, so any slice reverts to a
  working legacy state.
- The legacy goto-binary **reader is retained permanently** after S4; removing
  it is a separate, later, explicitly-approved decision.
- S5 (field removal) is the only irreversible step and runs only after S4's
  round-trip + cross-check evidence is green on the full corpus.

## Recommended order

```
S0a util+goto+symex+pointer   →  S0b clang   →  S0c python   →  S0d solidity+jimple+rest
S1  IREP2 cache + cross-check  (CENTRAL EVIDENCE GATE)
S2  pipeline readers → type2()/value2()
S3  writers → set_*(IREP2); kill contracts B6 round-trips
S4  flip native storage + serialization seam  (no format change)
S5  cleanup: drop legacy field + dead migrate_*_back
```

Estimated ~10 reviewable PRs. S0 (encapsulation) is the bulk of the diff but the
lowest risk; S1 is the linchpin evidence; S4 is the highest-risk single step.
