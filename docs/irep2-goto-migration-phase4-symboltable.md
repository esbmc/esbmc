---
title: "Phase 4 work breakdown — symbol table (symbolt) migration to IREP2"
status: draft
date: 2026-05-22
parent: docs/irep2-goto-migration-plan.md
tracking-issue: esbmc/esbmc#4715
---

# Phase 4 expanded: migrate `symbolt::value`/`type` to IREP2

Phase 4 is the **linchpin** (B2): it stores `typet type` and `exprt value`
(`src/util/symbol.h:15-16`), which every frontend writes and the whole goto
pipeline reads. Migrating it unblocks the forced B6 round-trips
(`contracts.cpp:1730,1896,…`; `add_race_assertions.cpp:35,85` — deferred PR 2.4).

It is also the **highest-risk** phase, so it gets the most sub-PRs and the
slowest, most defensive sequencing. The guiding principle: **the legacy fields
remain the source of truth until the very end; IREP2 fields are derived and
continuously cross-checked against them.** No frontend behaviour changes until
correctness on the read side is proven.

## Why the ordering is forced (ground truth)

1. **Writes cannot be intercepted today.** `contextt::find_symbol` returns a
   mutable `symbolt*` (`src/util/context.cpp:47-52`) and callers mutate
   `sym->value = …` / `sym->type = …` directly. There is no setter chokepoint.
   → **We must encapsulate before we can shadow-sync.** (PR 4.1 precedes 4.2.)
2. **The dual-role of `value` is keyed off `type`, not `value`.** A symbol's
   `value` is a function body iff `type.is_code()` (`goto_convert_functions.cpp:107,113`),
   otherwise it is a constant initializer (`python_converter.cpp:703`:
   `static_lifetime && !value.is_nil() && !type.is_code()`). Nothing checks
   `value.type() == type`. → preserve the `is_code` discriminator **exactly**;
   it becomes `is_code_type2t` under IREP2. This is the known
   `static_lifetime` + `symbol.value` hazard — guard it with a regression.
3. **Serialization is opaque and couples to Phase 5.** `to_irep` does
   `dest.type() = type; dest.symvalue(value)` (`symbol.cpp:98-99`); `from_irep`
   casts back (`:135-136`). `reference_convert` dedup
   (`irep_serialization.cpp:31-90`) is purely structural. → leave (de)serialization
   on legacy `irept` until Phase 5; the IREP2 fields are non-persistent and
   rebuilt from legacy on load. (PR 4.6 is the only one touching serialization,
   and it is the Phase-4/Phase-5 seam.)

## Shadow-field model

```
            ┌──────────────── symbolt ────────────────┐
 frontends  │  typet  type   (source of truth)  ◄──────┼── written by ~56 sites
   write ──►│  exprt  value  (source of truth)  ◄──────┼── written by ~31 sites
            │                                           │
            │  type2tc  type2  (DERIVED, synced) ───────┼─► read by goto pipeline
            │  expr2tc  value2 (DERIVED, synced) ───────┼─►   after PR 4.3
            └───────────────────────────────────────────┘
   sync point: a single setter (PR 4.1) recomputes type2/value2 via migrate_*.
   debug builds assert  migrate_back(type2) ≡ type  on every read (PR 4.2).
```

The legacy field stays authoritative through PR 4.5. PR 4.6 flips the source of
truth; PR 4.7 deletes the legacy fields (gated on Phase 5 owning serialization).

---

## PR 4.0 — `migrate_symbol` helper + invariant assertions (no behaviour change)

**Scope:** `src/util/migrate.{h,cpp}`, `src/util/symbol.{h,cpp}`.

**Edits:**
1. Add `void migrate_symbol(const symbolt &, /*out*/ type2tc &, expr2tc &)`
   centralising the field-by-field `migrate_type`/`migrate_expr` that callers
   currently open-code. Single tested conversion path.
2. Add a debug-only `symbolt::check_invariants()` documenting+asserting the
   discriminator: if `type.is_code()` then `value.is_nil() || value.is_code()`;
   if `static_lifetime && !type.is_code()` then `value` is a value-expr not a
   `codet`. Call it from `symbolt::show`/in unit tests only (no hot-path cost).
3. Unit test in `unit/util/`: round-trip a representative symbol set
   (function body, struct global initializer, scalar initializer, nil value,
   Python static-lifetime global) through `migrate_symbol` and assert
   `migrate_*_back` recovers a structurally-equal legacy field.

**Behaviour:** none — pure addition. **Gate:** unit test green; full build green.

---

## PR 4.1 — Encapsulate `symbolt::type`/`value` behind accessors (mechanical)

**Scope:** rename `symbolt::type`→`type_` / `value`→`value_` (private-by-convention)
and add `type()`/`value()` (const + mutable-set) accessors; mechanically update
all ~193 referencing files. **No shadow field yet** — accessors just return the
field. Purely mechanical, uniform diff = easy review.

**Split into reviewable sub-PRs by tree** (each independently compiles & passes):
- 4.1a `src/util/`, `src/goto-programs/`, `src/goto-symex/`, `src/pointer-analysis/`
- 4.1b `src/clang-c-frontend/`, `src/clang-cpp-frontend/`
- 4.1c `src/python-frontend/` (~290 access sites — largest)
- 4.1d `src/solidity-frontend/` (~187 sites; 24 value-writes in
  `solidity_convert_{decl,call,contract}.cpp`)
- 4.1e `src/jimple-frontend/`, remaining `unit/` and `src/esbmc/`

**Mechanics:** the mutable setter must be a method (`set_value(exprt)`,
`set_type(typet)`), *not* a returned reference, so PR 4.2 can hook it. Reads
keep returning `const typet&`/`const exprt&`. In-place edits like
`sym->value.operands()...` get rewritten to read-modify-`set` (these are rare;
flag each in review).

**Behaviour:** none. **Gate per sub-PR:** zero goto-binary diff on full corpus;
full regression green; clang-format clean.

---

## PR 4.2 — Add shadow fields + sync in setter + read-side assertions

**Scope:** `src/util/symbol.{h,cpp}`.

**Edits:**
1. Add `type2tc type2_; expr2tc value2_;`.
2. `set_type`/`set_value` recompute the shadow via `migrate_type`/`migrate_expr`
   (and clear to `type2tc()`/`expr2tc()` on nil, mirroring `value.make_nil()`).
3. `swap`/`clear`/`from_irep` keep both in sync (`from_irep` recomputes shadow
   after setting legacy — `symbol.cpp:135-136`).
4. Add `type2()`/`value2()` const accessors (IREP2 view) — not yet used by
   readers.
5. **Debug-build cross-check:** on every `type2()`/`value2()` read, assert the
   shadow round-trips to the legacy field (`migrate_type_back(type2_) ≡ type_`).
   This is the differential self-check (§5.3 of the master plan).

**Behaviour:** none (shadow is write-only-derived, read by nobody yet).
**Gate:** the cross-check assertion **never fires** across the full corpus +
SV-COMP subset; zero goto-binary diff; dual-solver agreement. *This is the
phase's central evidence gate* — it proves migrate is lossless on every symbol
ESBMC actually constructs.

---

## PR 4.3 — Switch goto-pipeline readers to the IREP2 view

**Scope:** `src/goto-programs/`, `src/goto-symex/` reader sites; frontends
untouched (still write legacy, synced).

**Edits:** migrate the consumers identified in the impact analysis to read
`value2()`/`type2()`:
- `goto_convert_functions.cpp:102,107,113,117` — `body_available`,
  `is_code_type2t` discriminator, `to_code_*2t(value2())` instead of
  `to_code(symbol.value)`.
- `goto_convert_functions.cpp:288,301` — argument `code_type2t` introspection
  (overlaps Phase 3 B3 work; coordinate).
- `namespace.cpp:39-43` value-following (read `value2()` when not code-typed).
- `context.cpp:128-141` merge logic — keep deciding on legacy `is_nil` for now
  (writes still legacy); only the *reads that feed the goto pipeline* move.
- the deferred PR 2.4 symbol-*reads* in `add_race_assertions`/contracts.

**Behaviour-preservation:** identical because shadow ≡ legacy (proven in 4.2).
The `is_code` → `is_code_type2t` swap is the one semantic-looking change;
back it with a focused regression on a function symbol and a same-named global.
**Gate:** zero goto-binary diff; dual-solver; the 4.2 cross-check still silent.

---

## PR 4.4 — Switch B6 writers to set IREP2 directly (kills the round-trips)

**Scope:** the passes that currently build `type2tc`/`expr2tc`, back-migrate,
then store legacy. With the setter syncing both ways we add an IREP2 setter
overload so they store natively.

**Edits:**
1. Add `set_type(type2tc)` / `set_value(expr2tc)` overloads that set the shadow
   directly and back-migrate to keep the legacy field valid for serialization.
2. Remove the round-trips:
   - `contracts.cpp:1730,1896,2145,2331,2369,2501,2540` (`migrate_type_back`).
   - `goto_atomicity_check.cpp:42`.
   - **Folds in deferred PR 2.4**: `add_race_assertions.cpp:35,85`
     (`new_symbol.set_type(array_type2tc(...))`, `set_value(gen_false_expr())`).
   - `assign_params_as_non_det.cpp` symbol writes.

**Behaviour-preservation:** `set_type(t2)` stores `t2` and derives
`migrate_type_back(t2)`; previously the code did `migrate_type_back(t2)` and the
setter (4.2) would derive `migrate_type(that)`. Net identity holds iff
`migrate_type ∘ migrate_type_back = id` on these nodes — **add a unit test
asserting that** for the specific node kinds these passes produce (array of
bool, snapshot struct/pointer types). **Gate:** zero goto-binary diff on
contracts + concurrency + data-race corpora; dual-solver.

---

## PR 4.5 — Tests: differential corpus + dual-role regression

**Scope:** `regression/`, `unit/`.

**Edits:**
1. Regression pinning the **dual-role discriminator**: a translation unit with
   (a) a function symbol, (b) a `static_lifetime` non-code global with a
   non-nil initializer, (c) a same-named local — assert ESBMC still emits the
   static initializer assignment and does *not* treat the body as data. This is
   the `gotcha_python_symbol_value_static_lifetime` guard, lifted to a test.
2. Python frontend: a `regression/python/` case exercising module-level globals
   (`static_lifetime=false`, `file_local=false`) to pin the interaction with
   the rw_set Python-global filter and const-prop snapshot behaviour. Run
   `scripts/check_python_tests.sh` on it.
3. Promote the 4.2 shadow cross-check to a CI THOROUGH/Linux job over the
   SV-COMP subset.

**Gate:** all green; tests fail if reverted to master (discriminating).

---

## PR 4.6 — Flip source of truth to IREP2 (Phase 4 / Phase 5 seam)

**Scope:** frontends (write IREP2 via the setter); `symbol.cpp` serialization.

**Edits:**
1. Migrate frontend write sites (PR 4.1 setters already in place) to call the
   IREP2 overloads; legacy field becomes the *derived* one.
   Sequence by cluster, smallest first: jimple → clang-cpp → clang-c → python →
   solidity (largest last, by then the pattern is proven).
2. `to_irep`/`from_irep`: until Phase 5 ships an IREP2-native goto-binary
   format, serialize from the derived legacy field (no on-disk format change).
   **This is the explicit hand-off point to Phase 5** — do not change the
   binary format here.

**Behaviour-preservation:** still gated on the (now-inverted) shadow cross-check
and zero goto-binary diff. **Gate:** full corpus diff = 0; dual-solver;
old goto-binaries still load (legacy reader unchanged).

---

## PR 4.7 — Cleanup (gated on Phase 5)

Remove the legacy `type_`/`value_` fields, the back-derivation in setters, and
the cross-check assertions — **only after Phase 5 owns IREP2-native
serialization** (otherwise `to_irep` still needs the legacy field). Until then,
PR 4.7 is blocked; record the dependency in #4715.

---

## Phase 4 exit criteria (all required)

- Shadow cross-check assertion silent across full corpus + SV-COMP subset
  (the lossless-migration proof).
- Zero goto-binary diff vs. Phase-0 baseline at every sub-PR.
- Dual-solver Bitwuzla + Z3 verdict **and** counterexample-digest agreement.
- The dual-role / `static_lifetime` regression (PR 4.5) passes and discriminates.
- `migrate_type_back`/`migrate_expr_back` call count in `src/goto-programs/`
  strictly decreased (the B6 round-trips from `contracts.cpp`,
  `goto_atomicity_check.cpp`, `add_race_assertions.cpp` are gone).
- `scripts/check_python_tests.sh` green (Python frontend touched).

## Suggested order & rollback

```
PR 4.0  migrate_symbol helper + invariant asserts   ← pure addition
PR 4.1  encapsulate type/value behind accessors      ← mechanical, split a–e by tree
PR 4.2  shadow fields + setter sync + read asserts    ← CENTRAL EVIDENCE GATE
PR 4.3  switch goto-pipeline readers to IREP2 view    ← revert restores legacy reads
PR 4.4  switch B6 writers to IREP2 (kills round-trips, folds in PR 2.4)
PR 4.5  tests (dual-role + Python globals)
PR 4.6  flip source of truth (Phase 4/5 seam — no format change)
PR 4.7  cleanup  ── BLOCKED on Phase 5
```

Rollback insurance: through PR 4.5 the legacy fields are authoritative, so any
sub-PR reverts cleanly to a working legacy state. PR 4.2's cross-check is the
trip-wire — if it ever fires, stop and fix `migrate_*` before proceeding rather
than weakening the assertion. PR 4.6 is the only irreversible-feeling step;
keep it behind the cross-check (now inverted) and a clean full-corpus diff
before merge, and do not let it touch the on-disk format — that is Phase 5's
deliberate, versioned decision.
