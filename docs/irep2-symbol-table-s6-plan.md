# S6 — Symbol-Table Migration: Cleanup and End-State Decision

Final slice of the IREP2 symbol-table migration (esbmc/esbmc#4715,
boundary B2). After V2 (#4741) the storage on `symbolt` is uniformly
IREP2-source-of-truth-with-lazy-legacy on both sides:

```cpp
private:
  mutable type2tc type_;                 // IREP2 source of truth
  mutable typet   legacy_type_cache_;
  mutable bool    legacy_type_valid_;
  mutable bool    type2_valid_;

  mutable expr2tc value_;                // IREP2 source of truth
  mutable exprt   legacy_value_cache_;
  mutable bool    legacy_value_valid_;
  mutable bool    value2_valid_;
```

The Phase 5 plan (#4732) and V-track plan (#4736) both deferred the
"drop legacy field / finalise ABI" question to this S6 plan. The
audit below grounds the decision.

## Caller surface, today

```
$ grep -rE "\.get_type\(\)|->get_type\(\)"   src/ unit/ | wc -l
447
$ grep -rE "\.get_value\(\)|->get_value\(\)" src/ unit/ | wc -l
205
$ grep -rE "\.set_type\(|->set_type\("       src/ unit/ | wc -l
152
$ grep -rE "\.set_value\(|->set_value\("     src/ unit/ | wc -l
155
```

Of those, only **7 sites** bind the return through a `const T&`
reference:

```
$ grep -rE "const typet\s*&\s*\w+\s*=\s*.*\.?get_type\(\)|...->get_type\(\)" ...
1  site

$ grep -rE "const exprt\s*&\s*\w+\s*=\s*.*\.?get_value\(\)|...->get_value\(\)" ...
6  sites
```

The remaining ~640 sites take the result transiently — through
`auto`, by-value `typet`/`exprt`, or directly inside a larger
expression (`sym.get_type().is_code()`). The reference-vs-by-value
ABI question therefore affects a **trivially small fraction of the
codebase**.

Chokepoint traffic in the migration layer:

```
$ grep migrate_symbol_type  src/ -r | grep -v migrate.cpp/h | wc -l   # 28
$ grep migrate_symbol_value src/ -r | grep -v migrate.cpp/h | wc -l   #  1
$ grep set_symbol_type      src/ -r | grep -v migrate.cpp/h | wc -l   # 17
```

Post-V2 these are trivial wrappers around `sym.get_type2()` /
`sym.get_value2()` / `sym.set_type(type2tc)`. They earned their keep
during the migration (single point for the NDEBUG cross-check, single
point for telemetry) — the question is whether to inline them now.

## Three end-state options

| Option | What changes | Cost | Benefit |
|---|---|---|---|
| **A — Drop legacy fields, return by value** (`typet get_type() const`) | Remove `legacy_type_cache_` / `legacy_value_cache_` and the `_valid_` bits. `get_type()` / `get_value()` back-migrate on **every** call. Source-incompatible at 7 binding sites. | `migrate_*_back` on every legacy read — hot paths (`is_code()`, `is_array()`, ...) pay each time. Caller churn at 7 sites. | Leaner `symbolt`: ~2 fewer fields per side. |
| **B — Keep legacy caches permanently; accept S6 as a documentation step** | Nothing in storage or API. Document the lazy-cache layout as the durable end state. Optionally tidy comments that still say "deferred to S6." | Zero. Two extra fields per symbol (≈1 `irept` shared pointer each post-COW). | Zero churn; preserves the lazy O(1) read on hot paths after cache warm-up. |
| **C — B plus inline the trivial chokepoints** | Replace `migrate_symbol_type(sym)` calls with `sym.get_type2()`, etc. Remove the now-redundant wrapper functions from `util/migrate.{h,cpp}`. | 28 + 17 + 1 = 46 sites to touch. Loses the single-point cross-check and the future-telemetry hook. | Slightly cleaner header; one less indirection. |

### Why Option B is recommended

1. **Storage cost is negligible.** `typet` and `exprt` are `irept`s with
   reference-counted sub-trees; the cache field after COW is one
   pointer plus the per-node header — not "another whole tree". On a
   100k-symbol program the storage delta is dominated by ordinary
   ESBMC data structures, not by this cache.

2. **Reads stay O(1) on hot paths.** Option A pays `migrate_*_back` on
   every `get_type()` / `get_value()` call. Symex and the goto-program
   passes call `sym.get_type().is_code()` and similar checks in tight
   loops; that's exactly the case the lazy cache was added to make
   cheap (#4730). Throwing it away post-V2 would regress those.

3. **API stability matters more than storage thrift.** The 640+ caller
   sites would not have to change under B; the seven `const T&`
   binding sites stay correct because the cache lifetime is the
   symbol's lifetime. Under A those seven sites become source-
   incompatible — small churn, but not zero, and for very little gain.

4. **The migration goal is already met.** B2's premise was "make the
   symbol-table storage IREP2-native." After V2 it is — `value_` and
   `type_` are the source of truth, the legacy fields are caches.
   That promise does not require *deleting* the caches; it requires
   IREP2 to be authoritative, which it is.

### Why Option C is rejected

The chokepoint functions (`migrate_symbol_type`,
`migrate_symbol_value`, `set_symbol_type`) earned their keep during
the migration:

- The NDEBUG cross-check inside each one runs on every real symbol the
  pipeline reads. It silently asserts the migration layer's
  losslessness. Inlining loses that.
- They are the canonical names a future developer searches for when
  asking "where does ESBMC convert a symbol's type between IREP1 and
  IREP2?". Inlining hides the answer behind one indirection deeper.
- The inlining cost (46 sites) is non-trivial, and the win
  ("one less function") is cosmetic.

Keep them. They cost nothing.

## Recommended path

**Option B, executed as one small commit.**

### S6 PR (single, small)

- Update the doc and comment scaffolding to reflect "B2 migration
  complete; lazy caches are the durable design."
- Specifically:
  - `docs/irep2-goto-migration-plan.md` — update the B2 status line.
  - `docs/irep2-symbol-table-migration-plan.md` — mark S5 / S6 done.
  - `docs/irep2-symbol-table-phase5-plan.md` — mark S5a / S5b done.
  - `docs/irep2-symbol-table-vtrack-plan.md` — mark V1 / V2 done; the
    V-track is closed.
  - `src/util/symbol.h` / `src/util/symbol.cpp` — remove the
    "deferred to S6" language; replace with "permanent lazy cache,
    end-state design."
  - `src/util/migrate.{h,cpp}` — comments updated similarly.
- No behavioural change. No new tests. The migration is complete with
  V2; this PR closes the book.

### Validation

- All existing unit suites stay green by construction (no code
  change).
- Sanity verdicts unchanged (no code change).
- Full build green.

## What S6 explicitly does **not** do

- **Does not drop the legacy `typet` / `exprt` fields.** They are the
  durable on-demand cache, not transitional.
- **Does not change `get_type()` / `get_value()` return type.** They
  remain `const T&` for the 640+ caller sites.
- **Does not inline `migrate_symbol_type` / `migrate_symbol_value` /
  `set_symbol_type`.** They earn their keep via the cross-check.

## Position relative to the larger IREP2 migration

The B2 boundary (`symbolt`) is closed by S6. The remaining boundaries
in `docs/irep2-goto-migration-plan.md` are independent and tracked
separately:

- B1 — frontend lowering (deliberately not migrated)
- B3 — `goto_functiont::type` (done in #4721)
- B4 — goto-binary serialization (no change; format opaque)
- B5 — rw_set internals (Phase 2 — `docs/irep2-goto-migration-phase2-rwset.md`)
- B6 — round-trip points (drained progressively across earlier PRs)

S6 closes B2. No further symbol-table migration work is planned.
