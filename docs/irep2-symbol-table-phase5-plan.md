# Phase 5 — Symbol-Table Source-of-Truth Flip

> **Status: complete.** S5a (#4735, with the lazy-storage fix in #4739)
> flipped the type side. The V-track (`irep2-symbol-table-vtrack-plan.md`)
> took the value side. The end-state ABI decision originally framed as
> S5b is recorded in `irep2-symbol-table-s6-plan.md`: legacy caches are
> retained permanently; `get_type()` / `get_value()` keep their
> `const T&` return.

Plan for the next step of the IREP2 symbol-table migration
(esbmc/esbmc#4715, Boundary B2). Phase 5 turns the IREP2 form of a
symbol's type into the **stored** field; the legacy `typet` becomes a
derived cache. The value side is deliberately left alone in this phase
— see *The `code_block2t` blocker* and *Value-side roadmap*.

This document is intentionally code-grounded: it cites the existing
state precisely, identifies the one architectural decision that has
to be made before code can be written, and lays out the slices that
follow once the decision is made.

## Status entering Phase 5 (what is already true)

| Step | PR | Effect |
|------|----|--------|
| S0 — encapsulate `symbolt::type` / `value` | [#4724](https://github.com/esbmc/esbmc/pull/4724) | Fields private; accessor surface mapped |
| S1 — `migrate_symbol_type` chokepoint + cross-check | [#4725](https://github.com/esbmc/esbmc/pull/4725) | Every read of a symbol's IREP2 type funnelled through one function |
| S1' — `migrate_symbol_value` chokepoint + guarded cross-check | [#4727](https://github.com/esbmc/esbmc/pull/4727) | Same for values; cross-check skips code bodies (see below) |
| S3 — `set_symbol_type` writer chokepoint | [#4728](https://github.com/esbmc/esbmc/pull/4728) | Every write of a symbol's IREP2 type funnelled through one function |
| S4a — audit & remove mutable getters | [#4729](https://github.com/esbmc/esbmc/pull/4729) | ~200 write sites routed through setters; compiler-enforced |
| S4b — add IREP2 cache to `symbolt`; chokepoints read it | [#4730](https://github.com/esbmc/esbmc/pull/4730) | `migrate_symbol_*` is now O(1); cache invalidated by every setter |
| S2/S3 closeout — last three bypass sites | [#4731](https://github.com/esbmc/esbmc/pull/4731) | 100 % of symbol type read/write traffic now flows through the chokepoint |

Concretely the current `symbolt` (after #4730) is:

```cpp
class symbolt {
  // ... public flags / id / mode ...
  const typet  &get_type()  const;           // returns legacy field
  const exprt  &get_value() const;           // returns legacy field
  const type2tc &get_type2()  const;         // lazy IREP2 cache
  const expr2tc &get_value2() const;         // lazy IREP2 cache
  void set_type(const typet &);              // legacy write; invalidates cache
  void set_type(const type2tc &);            // IREP2 write; legacy derived
  void set_value(const exprt &);             // legacy write; invalidates cache
  // ... swap / clear / to_irep / from_irep ...
private:
  typet  type;       // <-- still the source of truth
  exprt  value;      // <-- still the source of truth
  mutable type2tc type2_cache;
  mutable expr2tc value2_cache;
  mutable bool    type2_valid;
  mutable bool    value2_valid;
};
```

The NDEBUG cross-check inside `migrate_symbol_type` /
`migrate_symbol_value` already exercises
`migrate_type(migrate_type_back(cached)) == cached` on every real symbol
the pipeline reads. After #4731, that cross-check covers 100 % of symbol
type traffic — the corpus-wide evidence the original plan
(`docs/irep2-symbol-table-migration-plan.md`) was built around.

## The `code_block2t` blocker

`migrate_expr_back` in `src/util/migrate.cpp:2486` is a switch over
`expr2t::expr_id`. The `default:` arm at line 3617 reads:

```cpp
default:
  log_error("Unrecognized expr in migrate_expr_back");
  abort();
```

`code_block2t` (and a number of other code-statement kinds —
`code_function_call2t`, `code_decl2t`, etc.) **are not in the switch**
in the back direction even though they are in the forward direction
(`migrate_expr`, called by goto-convert). The asymmetry is documented
in `unit/util/migrate.test.cpp`:

> Individual code statements round-trip (goto2c back-migrates
> instructions one at a time). Note: `migrate_expr_back` deliberately
> does NOT support a whole `code_block2t` — a function body's legacy
> codet block is migrated forward only (it is the source of truth),
> so the symbol-table migration never needs to reconstruct a block
> from IREP2.

That assumption — "the symbol-table migration never needs to
reconstruct a block from IREP2" — is exactly what S5 has to revisit.

### Why this blocks the value-side flip

If `symbolt::value` becomes IREP2-authoritative, then every caller of
`sym.get_value()` that today receives `const codet &` (a function body)
needs the legacy form derived from the IREP2 store via
`migrate_expr_back`. Seven sites read `get_value()` and use it as a
body:

| Site | Use |
|---|---|
| `esbmc/esbmc_parseoptions.cpp:3014` | `to_code(fn->get_value())` to feed `goto_convert` |
| `goto-programs/goto_main.cpp:31` | `goto_convert(to_code(s->get_value()), ...)` |
| `goto-programs/goto_convert_functions.cpp:106,112,116` | body availability / kind / pretty-print |
| `python-frontend/converter/converter_funcall.cpp:54,56` | call-site body inspection |

Beyond those, `goto_convert_functions.cpp:422,433` walks the body via
`collect_expr(s.get_value(), ...)`, and `to_irep` (the goto-binary
write seam) reads `value` directly. None of these would currently
survive a flip that derived legacy from IREP2 storage.

### Why it does **not** block the type-side flip

`code_type2t` (function signatures) round-trips cleanly:
`migrate_type_back` handles it (it has been in the back direction since
the dual-role audit, exercised by the chokepoint cross-check on every
function symbol). `migrate.test.cpp` includes a `code_type2tc` round-trip
test case for exactly this reason.

So the type-side flip is unblocked; the value-side flip is not, until
either the back-migration of bodies is implemented, or value storage is
explicitly excluded from the flip.

## Three design options

| Option | What changes | What stays | Risk |
|---|---|---|---|
| **A — symmetric flip** | Both `type` and `value` become IREP2-stored; legacy derived. Extend `migrate_expr_back` to handle `code_block2t` and every code-statement sub-kind. | Nothing | High — substantial new migration code, no existing test infrastructure for it |
| **B — type-only flip** | `type` becomes IREP2-stored; legacy `typet` derived. `value` stays legacy-stored exactly as today. | All seven `to_code(get_value())` call sites | Low — type round-trip is already proved by the chokepoint cross-check |
| **C — symmetric flip with carve-out** | Both fields IREP2-stored except function bodies, which retain a legacy `exprt` next to the IREP2 form. `is_code` branches storage. | Body callers | Medium — adds storage branching to `symbolt`; harder to reason about than B |

### Why Option B is the recommended path

1. **The corpus-wide cross-check is already silent for types.** Every
   real symbol the pipeline reads has gone through
   `migrate_type(migrate_type_back(cached)) == cached` in every NDEBUG
   build since #4725. After #4731, that's 100 % coverage of symbol type
   traffic. The hard part of S4 (the *plan*'s S4, "flip source of
   truth") is already evidenced.
2. **Half the legacy `irept` storage on `symbolt` disappears.**
   `typet` is the bulkier of the two for most non-body symbols.
3. **The value side keeps its existing semantics.** Seven body-reading
   call sites stay correct; no migration of code statements needed.
4. **It is reversible and slice-able.** S5a (the flip) and S5b (drop
   the legacy field) are independent PRs; either can be reverted
   without touching the other.
5. **Option A's prerequisite work is unscoped.** Adding `code_block2t`
   to `migrate_expr_back` means picking back-direction representations
   for ~20 statement kinds (assigns, calls, decls, returns, gotos,
   labels, atomic-begin/end, switch, ...). That's a project, not a
   slice. Doing it as a precondition for S5 would make S5 unboundedly
   large.

### Why not Option C

The carve-out introduces a "tagged storage" pattern on `symbolt` —
`type` is IREP2, `value` is conditionally IREP2 or legacy depending on
`is_code(type)`. The branching pushes complexity into every accessor
and every serializer. The win over Option B (slightly fewer legacy
fields on function symbols) is not worth the bookkeeping cost.

## S5 slices (under Option B)

Each row is one PR. They share a branch chain off master in the order
listed, but each is independently revertible.

### S5a — flip type storage; legacy becomes derived cache

**Effect.** `symbolt::type` (legacy `typet`) is removed as a stored
field; replaced by `type2tc type_` (the IREP2 source of truth) and
`mutable typet legacy_type_cache_` + `mutable bool legacy_type_valid_`
(populated lazily by `get_type()` via `migrate_type_back`). This is
S4b's design with the roles reversed.

**Accessor semantics.**

| Method | Before (post-S4b) | After (S5a) |
|---|---|---|
| `get_type()` | returns ref to legacy field | returns ref to lazy legacy cache |
| `set_type(const typet&)` | writes legacy; invalidates IREP2 cache | `type_ = migrate_type(t)`; legacy cache populated eagerly with `t` |
| `set_type(const type2tc&)` | writes IREP2 cache; derives legacy via `migrate_type_back` | `type_ = t`; invalidates legacy cache |
| `get_type2()` | returns lazy IREP2 cache | returns `type_` directly (no cache needed) |

**Serialization.**
- `to_irep`: derive legacy via `get_type()` (which migrates back on
  first call), assign to `dest.type()`. **Same on-disk format.**
- `from_irep`: assign `src.type()` to legacy cache, eagerly forward-
  migrate to `type_`. Old binaries still load.

**Gate.** Differential goto-binary diff = 0 on the corpus harness
(`scripts/irep2-migration/`); cross-check silent; sanity verdicts;
unit tests; dual-solver agreement on a smoke set.

**Risk areas.**
- Lifetime of `const typet&` returned from `get_type()`: the cache
  must outlive the reference. Mutable cache populated lazily inside
  the const method, sentinel-guarded — standard pattern, already used
  in S4b for the IREP2 cache.
- Performance: `migrate_type_back` runs on first `get_type()` call per
  symbol; warm after that. Cost should be at-or-below the per-read
  `migrate_type` we eliminated in S2.
- COW aliasing of `type2tc`: writes that previously mutated a `typet`
  sub-tree in place must now go through the setter. S4a already
  enforced this for the legacy write path — the constraint is the same.

### S5b — drop the legacy `typet` cache (open question)

This is the ABI decision flagged in #4730's PR description.

**Two sub-options.** Pick at the time S5a lands; either is feasible.

| Sub-option | `get_type()` signature | Effect |
|---|---|---|
| **B-keep-ref** | `const typet &get_type() const` (unchanged) | Keep `mutable typet legacy_type_cache_` as a permanent on-demand cache. Same source-level API for callers. Storage stays at *one* `typet` per symbol that has had `get_type()` called. |
| **B-by-value** | `typet get_type() const` | Drop the cache. Every `get_type()` call back-migrates. Source-incompatible: callers binding `const typet&` must change. Saves the cache field. |

`B-keep-ref` is the lower-friction default; `B-by-value` only becomes
attractive if we want a leaner `symbolt`. Defer the choice; S5a is
already useful without it.

### S5c (deferred) — value-side roadmap

Out of scope for Phase 5 under Option B. Sketched here for forward
reference only.

1. **V1 — audit `get_value()` body callers.** Classify the seven sites
   above plus `collect_expr` walks: which actually need a `codet`
   versus which could read an `expr2tc` via `get_value2()`.
2. **V2 — IREP2-native body traversal.** For sites that only need to
   walk, replace `collect_expr(s.get_value(), ...)` with an `expr2tc`
   walker (the goto-symex side already has these — reuse rather than
   adding back-migration cases).
3. **V3 — decide.** Either extend `migrate_expr_back` for the body
   kinds and flip value storage, or commit to value staying legacy
   permanently.

V3 is a genuine architectural decision (do we want `symbolt::value`
to be IREP2 at all?), not just an implementation choice. Defer to a
separate plan once V1/V2 surface the real cost.

## Validation strategy

Same shape as S4a/S4b:

1. **Cross-check.** The S1 NDEBUG round-trip assertion stays in place
   through S5a. After the flip, the assertion changes orientation
   (`migrate_type_back(migrate_type(legacy)) == legacy` — i.e. the
   lazy cache is consistent with what would be re-derived). Same
   property, same fire condition.
2. **Differential goto-binary diff.** Run
   `scripts/irep2-migration/capture_goto_baseline.sh` on master,
   `diff_goto_baseline.sh` on the S5a branch. Zero diff per slice,
   with the documented `__file__` / astgen-path non-determinism
   filters.
3. **Unit tests.** `migratetest`, `irep2test`, `symboltest`,
   `base_typetest` all stay green. Extend `migrate.test.cpp` if any
   new round-trip property is asserted.
4. **Sanity verdicts.** C assert SUCCESSFUL; C OOB FAILED; data-race
   SUCCESSFUL on safe, FAILED on real W/W race (scalar and array
   indexed — the latter touches `add_race_assertions.cpp` rewritten
   in #4731).
5. **Dual-solver.** Bitwuzla + Z3 verdict agreement on the smoke set.
6. **Python sanity.** `scripts/check_python_tests.sh` for any
   Python-touching slice.
7. **Read-from-old-binary.** Save a goto-binary under master; load it
   under the S5a branch; round-trip through `from_irep` / `to_irep`;
   assert byte-identical on re-write.

## Risk register

| Risk | Mitigation |
|---|---|
| `to_irep` accidentally changes the goto-binary format | Format fact 1 (opaque) — the legacy `typet` written to disk is whatever `get_type()` returns. The lazy cache returns the same `typet` the legacy field used to. Read-from-old-binary test gates this. |
| Lifetime of `const typet&` returned from `get_type()` | Cache lifetime tied to the `symbolt`. Same pattern as S4b's `type2tc` cache. |
| Performance regression from `migrate_type_back` on cache miss | Per-symbol one-shot cost; warm after first `get_type()` call. Measure symex on a heavy benchmark before/after S5a; abort if > 2 % regression. |
| COW aliasing — in-place mutation of a `typet` returned by `get_type()` | Already prevented by S4a (`get_type()` returns `const typet&`); no caller can mutate. |
| The dual-role hazard (function symbol + same-named `static_lifetime` global) | Add a regression test pinning the `is_code` discriminator before S5a (see `docs/irep2-symbol-table-migration-plan.md` § Risk). |

## Rollback

- S5a is a single-file change to `src/util/symbol.{h,cpp}` plus
  minor edits to `src/util/migrate.cpp`. Revert = single PR revert.
- The legacy `typet` interface (`get_type()` returning `const typet&`)
  stays binary-compatible under both sub-options of S5b.
- The S1 cross-check is the canary: if it fires on any real symbol,
  revert that PR.
- The legacy goto-binary reader is retained permanently — no on-disk
  format change.

## Recommended order

1. **Pre-S5a** — add the dual-role regression test (function + global
   with the same name; assert `is_code_type2t(get_type2())` parity
   with `get_type().is_code()`).
2. **S5a** — type-side flip (this is the substantial PR).
3. **S5b** — drop the legacy `typet` cache or keep it (pick a
   sub-option once S5a is in master a week or two).
4. **V-track (separate plan)** — value-side audit & decision.

This phase ends Phase 5 under Option B; further migration (value side,
B6 round-trips) is tracked separately.
