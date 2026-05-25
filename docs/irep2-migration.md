# IREP2 migration — retrospective

This document records the migration of ESBMC's goto-program pipeline
from the legacy string-based representation (`irept` / `exprt` /
`typet` / `codet`) to the typed IREP2 representation (`expr2tc` /
`type2tc`). The migration completed in 2026 under tracking issue
[#4715](https://github.com/esbmc/esbmc/issues/4715); this is the
single canonical record of what was done and why.

The detailed staged plans that drove the execution lived in
`docs/irep2-*.md` while the work was in flight; they were
consolidated into this retrospective once everything landed. The
commit log (`git log --all --grep "B2\\|IREP2\\|esbmc/esbmc#4715"`)
preserves every per-stage plan and decision rationale.

## Headline

`symbolt::type` and `symbolt::value` are now stored as `type2tc` /
`expr2tc` natively, with the legacy `typet` / `exprt` retained as
**permanent on-demand lazy caches**. The deep verifier (symex,
solvers, pointer-analysis) was already IREP2 before the migration
started; the migration drained the legacy storage from the symbol
table — the pipeline's central data structure — and from the
analysis passes that orbited it.

On-disk goto-binary format: **unchanged**. Old binaries still load.

## Why this mattered

The legacy IREP is a string-keyed tree (`irept`) where types and
sub-expression kinds are encoded as `irep_idt`s parsed at every
access. IREP2 uses typed C++ classes whose kind is fixed at
construction and whose fields are accessed by name. Two consequences
drove the migration:

1. **Determinism.** Hash-based containers keyed off IREP2 nodes have
   stable, type-aware hashing; the legacy `irept` hash relies on
   string interning and is more fragile under concurrent access.
2. **Soundness via the type system.** "What kind is this expression?"
   becomes a compile-time question for IREP2 (`is_constant_int2t(e)`)
   versus a runtime string compare for legacy
   (`e.id() == "constant"`). Pipeline code that takes a wrong path on
   a misclassified expression fails at compile time after migration,
   not in symex.

ESBMC is a verifier — behavioural correctness outranks every other
consideration. The migration was therefore behaviour-preserving at
every step: differential goto-binary diff zero, dual-solver verdict
agreement, no on-disk format change.

## The six boundaries

The original plan identified six legacy boundaries:

| # | Boundary | Status |
|---|----------|--------|
| B1 | Frontend → goto input (frontends emit legacy AST) | **Deferred indefinitely.** The `migrate_*` seam at goto-convert is the durable boundary; migrating the frontends themselves would be a much larger project with marginal verifier-side benefit. |
| B2 | **Symbol table** (`symbolt::value` / `::type`) | **Done.** Detailed below — the foundational, highest-risk boundary. |
| B3 | `goto_functiont::type` | **Done in [#4721](https://github.com/esbmc/esbmc/pull/4721).** Function-signature type became `type2tc`. |
| B4 | Goto-binary serialization (`irept`-based on-disk format) | **No change planned.** The on-disk format is the project's external contract; the symbol-table `to_irep` / `from_irep` bridges via the lazy legacy caches. Old binaries still load; legacy reader retained permanently. |
| B5 | `rw_set` data-race pass | **Done in [#4718](https://github.com/esbmc/esbmc/pull/4718) (internals), [#4719](https://github.com/esbmc/esbmc/pull/4719) (w_guardst), [#4748](https://github.com/esbmc/esbmc/pull/4748) (focused tests).** Self-contained island; data-race pass operates in IREP2 throughout. |
| B6 | Round-trip points (passes that build `exprt`, migrate, then store) | **Drained.** Symbol-write round-trips were eliminated as B2 closed (the chokepoint `set_symbol_type` stores IREP2 natively). The residual `migrate_*_back` calls in `goto_check.cpp` / `goto_atomicity_check.cpp` are not round-trips — they are legitimate adapters feeding legacy APIs (`c_expr2string`, deref helpers) and would be eliminated only by migrating those callees, which is independent work. |

## B2 — symbol table — the foundational step

`symbolt` is the data structure every frontend writes to and every
analysis pass reads from. Migrating it required the most care; it
took 17 PRs over 16 stages. The shape of the final design is in
`src/util/symbol.{h,cpp}`:

```cpp
private:
  // Both representations are storage; the setter that wrote last
  // marks its side valid and invalidates the other. The reader on
  // the invalidated side lazily derives via the migration layer.
  mutable type2tc type_;                 // IREP2 source of truth
  mutable typet   legacy_type_cache_;    // lazy back-migrated cache
  mutable bool    legacy_type_valid_;
  mutable bool    type2_valid_;

  mutable expr2tc value_;                // IREP2 source of truth
  mutable exprt   legacy_value_cache_;   // lazy back-migrated cache
  mutable bool    legacy_value_valid_;
  mutable bool    value2_valid_;
```

`get_type()` / `get_value()` return `const T&` to the legacy cache
(populated on first access). `get_type2()` / `get_value2()` return
the IREP2 source directly. Setters write one side and invalidate the
other; readers lazily derive on first access.

### Why lazy on both sides, not eager forward-migrate

S5a (the type-side flip, [#4735](https://github.com/esbmc/esbmc/pull/4735))
originally made `set_type(const typet&)` forward-migrate eagerly,
populating both sides at write time. The full Python regression suite
caught this in CI: the python frontend builds tmp symbol types whose
internal sub-expressions sometimes have no `typet` set
(`minus_exprt(lhs, rhs)` whose result `typet` is default-constructed
with empty id). The eager `migrate_type` walked those holes and
constructed IREP2 arithmetic nodes with empty result types, tripping
`assert_arith_2ops_consistency` at construction.

Hotfix [#4739](https://github.com/esbmc/esbmc/pull/4739) restored
lazy migration on both sides. The principle: **never forward-migrate
on write; let the migration happen only when a reader actually asks
for the other form**. This tolerates latent legacy holes as long as
no pipeline path reads the IREP2 form of the affected symbol — which
in practice is exactly what happens. V2 (the value-side flip,
[#4741](https://github.com/esbmc/esbmc/pull/4741)) was designed lazy
from day one.

### Why retain the legacy caches permanently

S6 ([#4742](https://github.com/esbmc/esbmc/pull/4742) plan,
[#4743](https://github.com/esbmc/esbmc/pull/4743) finalise) framed
the end-state as a choice:

- **Option A** — drop the legacy fields; `get_type()` / `get_value()`
  return by value, paying `migrate_*_back` on every call. Source-
  incompatible at the seven `const T&` binding sites (out of 640+
  caller sites in `src/`).
- **Option B** — keep the lazy caches permanently.

Audit data settled it. The accessor cost matters: hot paths call
`sym.get_type().is_code()` and similar in tight loops; paying a
back-migration on every call would regress symex throughput
materially. The storage cost is negligible (one `irept` shared
pointer per side, post-COW). Option B preserves the O(1) read
property the cache was added for in
[#4730](https://github.com/esbmc/esbmc/pull/4730) (S4b).

### The migration chokepoints

`util/migrate.h` exposes three named chokepoints:

```cpp
type2tc migrate_symbol_type(const symbolt &sym);
void    migrate_symbol_value(const symbolt &sym, expr2tc &dest);
void    set_symbol_type(symbolt &sym, const type2tc &t);
```

Post-migration these are thin wrappers around `sym.get_type2()` /
`sym.get_value2()` / `sym.set_type(t)`. They are deliberately retained
rather than inlined: each runs an `NDEBUG` round-trip cross-check
asserting `migrate_*(migrate_*_back(form)) == form` on every real
symbol the pipeline reads. The check is the strongest losslessness
evidence the migration produces; inlining the wrappers would lose it.

## V-track — closing the back-migration coverage gap

Phase 5 (the type-side flip plan) tentatively rejected migrating the
value side, asserting that "the symbol-table migration never needs to
reconstruct a code_block from IREP2." The audit done for the V-track
plan refuted this: of 109 expr2t kinds, **104 were already covered**
by `migrate_expr_back`; only five were missing
(`code_block2t`, `code_cpp_catch2t`, two `code_cpp_throw_decl*2t`
forms, and `pointer_capability2t`). V1
([#4737](https://github.com/esbmc/esbmc/pull/4737)) added the missing
arms with unit-test round-trip gates; V2 then flipped the value
storage.

The crucial idea: V1 was dead code in the pipeline until V2 wired it
in. Adding the back-migration arms with round-trip tests as a
standalone PR meant V2 had zero behaviour-change risk on the
back-migration coverage axis — V1 had already proven the property in
isolation.

## Validation strategy

Every B2 stage was gated on the same four checks:

1. **Build green** (esbmc + jimple + the regression tests' direct
   targets).
2. **Unit-test suites** for the affected layer (`symboltest`,
   `migratetest`, `irep2test`, `base_typetest`).
3. **Sanity verdicts** on a small reproducer set: C `assert(x==1)`
   SUCCESSFUL, C out-of-bounds FAILED, W/W data-race on a shared int
   FAILED.
4. **The NDEBUG cross-check** running silently across the corpus on
   every real symbol read — the central piece of evidence that the
   migration layer is lossless on the kinds the pipeline actually
   touches.

The full Python regression suite caught one latent bug
([#4737](https://github.com/esbmc/esbmc/pull/4737)'s CI surfaced the
S5a eager-migration regression). After that, the lazy-by-design
principle prevented any recurrence on V2.

## The PR sequence — for the record

| Stage | PR | What |
|-------|----|------|
| Phase 0 — harness | #4717 | Differential goto-binary diff harness (`scripts/irep2-migration/`). |
| B5 phase 2.1 | #4718 | `rw_sett::entryt` storage migrated to IREP2. |
| B5 phase 2.2 | #4719 | `w_guardst` expression building migrated. |
| Phase 4 groundwork | #4722 | Migrate round-trip unit tests pinning IREP2 ↔ legacy idempotence. |
| B2 plan (docs) | #4723 | Code-grounded B2 execution plan. |
| B2 S0 | #4724 | Encapsulate `symbolt::type`/`value`; fields private. |
| B2 S1 | #4725 | `migrate_symbol_type` chokepoint + cross-check. |
| B2 S1' | #4727 | `migrate_symbol_value` chokepoint with guarded cross-check. |
| B2 S3 | #4728 | `set_symbol_type` writer chokepoint. |
| B2 S4a | #4729 | Audit writes; remove mutable getters (~200 sites). |
| B2 S4b | #4730 | IREP2 lazy cache on `symbolt`. |
| B2 S2 closeout | #4731 | Last three bypass sites converted. |
| Phase 5 plan (docs) | #4732 | Type-side flip plan, Option B sign-off. |
| Pre-S5a tests | #4733 | Dual-role discriminator-parity unit tests. |
| **B2 S5a** | **#4735** | **Type-side source-of-truth flip.** |
| V-track plan (docs) | #4736 | Value-side audit + recommendation. |
| V1 | #4737 | `migrate_expr_back` coverage for the 5 missing kinds. |
| S5a hotfix | #4739 | Lazy-on-write (eager forward-migrate regression). |
| Pre-V2 tests | #4740 | Value-side discriminator-parity unit tests. |
| **B2 V2** | **#4741** | **Value-side source-of-truth flip.** |
| S6 plan (docs) | #4742 | End-state Option B decision. |
| S6 finalise | #4743 | Docs + comment cleanup; B2 closed. |
| B5 phase 2.3 | #4748 | Focused rw_set unit + regression tests. |

## Surface ratio at the close

Line counts of type-name mentions in `src/`:

| Tree | IREP2 | Legacy | IREP2 share |
|------|------:|-------:|------------:|
| `solvers/` | 860 | 8 | 99 % |
| `pointer-analysis/` | 504 | 10 | 98 % |
| `goto-symex/` | 1162 | 26 | 98 % |
| `goto-programs/` | 1007 | 474 | 68 % |
| `util/` (incl. `symbolt`) | 1262 | 2247 | 35 % |
| frontends (clang-c, clang-cpp, python, solidity, jimple) | 8 | 5367 | <1 % |

The deep verifier core is uniformly IREP2. The symbol table is
IREP2-source-of-truth with lazy legacy caches. The frontends are
intentionally legacy; the `migrate_*` seam at goto-convert is the
durable boundary.

## Where to look in the code

| What | Where |
|------|-------|
| Symbol storage layout | `src/util/symbol.h` |
| Lazy cache implementation | `src/util/symbol.cpp` |
| Migration layer (forward, back, chokepoints) | `src/util/migrate.{h,cpp}` |
| `migrate_expr_back` switch (all 109 kinds covered after V1) | `src/util/migrate.cpp` (~line 2486 onwards) |
| IREP2 kind manifest | `src/irep2/expr_kinds.inc`, `src/irep2/type_kinds.inc` |
| Round-trip unit tests | `unit/util/migrate.test.cpp`, `unit/util/symbol.test.cpp` |
| rw_set IREP2 paths + tests | `src/goto-programs/rw_set.{h,cpp}`, `unit/goto-programs/rw_set.test.cpp` |
| Differential harness | `scripts/irep2-migration/` |

## What is *not* changing

For future readers wondering "should I migrate X next?" — the
following were considered and explicitly left:

- **Frontend ASTs** stay legacy `irept`. The clang / python / solidity
  / jimple frontends all emit legacy `exprt` / `typet`; the
  `migrate_expr` seam at goto-convert is the right permanent boundary.
- **Goto-binary on-disk format** stays `irept`-based. The external
  contract with stored binaries is more valuable than format
  modernization. The legacy reader is retained permanently.
- **`get_type()` / `get_value()` return type** stays `const T&`. The
  audit (#4742) showed by-value returns would regress hot paths
  measurably for very little gain.
- **The `migrate_symbol_*` chokepoints** stay as named functions.
  Inlining them is a 46-site rewrite for cosmetic gain; their named
  presence preserves the cross-check and gives future debuggers a
  greppable seam.

The migration is **done**. Future incremental work — eliminating
`c_expr2string`'s legacy dependency, or revisiting one of the
explicit non-goals above — should live under its own focused tracking
issue, not under the umbrella that closed with this retrospective.
