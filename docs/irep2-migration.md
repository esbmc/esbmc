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

---

# Part II — `util/` analysis & helper migration (forward plan)

> **Status: planning.** Part I above is a closed retrospective of the
> symbol-table boundary (B1–B6). This part is the *focused follow-on*
> that the retrospective anticipated ("future incremental work … should
> live under its own focused tracking issue"). It is a plan, not a
> record: nothing here has landed yet. Tracking issue: **to be filed**.

## 1. Scope and relationship to Part I

Part I drained the legacy IR out of the **symbol table** — the
pipeline's central data structure — and the analysis passes that orbit
it (`rw_set`, `goto_functiont::type`). The deep verifier core
(`goto-symex`, `solvers`, `pointer-analysis`) was already IREP2.

What Part I left untouched is the **`src/util/` analysis-and-helper
layer**: the simplifiers, type/cast builders, equality engines,
pretty-printers, expression helpers and the legacy representation
itself. The retrospective's own surface-ratio table put `util/` at
**35 % IREP2 / 65 % legacy** at close — the lowest share of any tree
outside the frontends. This part audits that residual surface and lays
out what can be migrated, what must be retained, and in what order.

The hard constraint is unchanged and non-negotiable: **ESBMC is a
verifier; every step must be behaviour-preserving** — differential
goto-binary diff zero, dual-solver verdict agreement, no on-disk format
change. A migration that is merely "cleaner" but shifts a single verdict
is a regression.

## 2. How this plan was derived (reproducibility)

Every classification below is grounded in the code at the commit this
document lands on, not in recollection. The method, re-runnable by any
reviewer:

- **Caller graph** — `grep -rln '#include.*<util/NAME.h>' src` per
  header, bucketed into *frontends* (`clang-c`, `clang-cpp`, `python`,
  `solidity`, `jimple` — the declared permanent legacy boundary),
  *gotoPrograms*, *core* (`goto-symex` ∪ `solvers` ∪
  `pointer-analysis`, already IREP2), *util*, *tests*, *other*.
- **IR per symbol** — read each file; a function is legacy if it
  takes/returns `exprt`/`typet`/`codet`/`irept`, IREP2 if
  `expr2tc`/`type2tc`.
- **irep2 equivalent exists?** — grep `src/irep2/irep2_utils.{h,cpp}`,
  `irep2_expr.h`, `irep2_type.h`, `c_types.h` for a twin.
- **Adversarial verification** — every load-bearing count below was
  re-checked by an independent pass instructed to *refute* it; the
  numbers here are the post-correction figures.

Counts are a snapshot; re-run `scripts/irep2-migration/migrate_census.sh
src/util` and the per-header `grep` before acting on any single phase.

## 3. The five migration tiers

Each `util/` component is assigned exactly one disposition:

| Tier | Meaning | Action |
|------|---------|--------|
| **ALREADY_IREP2** | Operates on `expr2tc`/`type2tc` today. | None. Confirm census shows no stray `migrate_*_back`. |
| **RETIRE_DEAD** | Legacy code with a complete IREP2 replacement and few/zero live callers. | Redirect stragglers, then delete (C-Dead proof). |
| **DUAL_API_THEN_DROP** | Already exposes both legacy and IREP2 overloads; core uses IREP2, frontends/stragglers use legacy. | Add an equivalence test, migrate non-frontend callers, drop the legacy overload. |
| **MIGRATE_CALLERS_FIRST** | Legacy and load-bearing; cannot move until its callers (mostly frontends) move. | Gate, don't migrate. Blocked on the frontend boundary. |
| **RETAIN_BOUNDARY** | *Is* the legacy representation, the on-disk format, or the migrate seam. | Keep permanently; validate it is **not** accidentally changed. |

## 4. Component inventory and classification

All file paths are under `src/util/` unless noted. Caller buckets are
*external* includers (the file's own `.cpp` excluded).

| Cluster | Files | Tier | Caller split (front / goto / core / util / other) | IREP2 equivalent | Disposition |
|---------|-------|------|---|---|---|
| **Legacy simplifier** | `simplify_expr.{cpp,h}`, `simplify_expr_class.h`, `simplify_utils.{cpp,h}` | **RETIRE_DEAD** | free `simplify(exprt&)`: **6 sites / 3 files**; `simplify_exprt` class: **1** (`c_typecast.cpp:919`) | `expr2t::simplify` / `simplify(expr2tc&)` (`irep2_utils.h:265`), 81 `do_simplify` overrides, 134 unit cases | Redirect 7 sites, delete 5 files |
| **IREP2 simplifier** | `expr_simplifier.cpp`, `expr_reassociate.{cpp,h}` | **ALREADY_IREP2** | — | self | Keep; only Class-B census cleanup (§7, Phase 2.3) |
| **base_type / type_eq** | `base_type.{cpp,h}`, `type_eq.{cpp,h}` | **DUAL_API_THEN_DROP** (+ `type_eq` RETIRE_DEAD) | legacy `base_type_eq(typet,…)`: **9 live**; `base_type_eq(exprt,exprt)` & `base_type(exprt&)`: **0**; `type_eq()`: **0** | `base_type_eqt::base_type_eq(type2tc/expr2tc)` already the core path | Drop 0-caller overloads + `type_eq` now; drop legacy `typet` overload after c_link/c_typecast/goto2c move |
| **C type/cast builders** | `c_types.{cpp,h}`, `c_typecast.{cpp,h}`, `c_sizeof.{cpp,h}`, `arith_tools.{cpp,h}` | **DUAL_API_THEN_DROP** | `c_typecast` non-frontend callers: **1** (`interval_domain.cpp`, already IREP2 overload) | `*_type2`/`get_*_type` siblings in `c_types.h`; `c_sizeof` legacy is a migrate shim | Add 13 missing `*_type2` siblings; migrate goto/symex `size_type`/`bool_type`/`char_type`; drop legacy after frontends |
| **Legacy expr helpers** | `expr_util.{cpp,h}` | **MIGRATE_CALLERS_FIRST** (dead helpers RETIRE_DEAD) | **103** includers (54 front / 28 core / 15 goto / 3 util / 3 other) | twins in `irep2_utils.h` (`gen_zero`, `gen_one`, `make_not`, `conjunction`/`disjunction`, `gen_true_expr`) | Delete dead helpers now; rest blocked on frontends |
| **Legacy printers** | `c_expr2string.{cpp,h}`, `cpp_expr2string.{cpp,h}`, `type2name.{cpp,h}` | **MIGRATE_CALLERS_FIRST** | **8** includers (4 front / 2 goto / 1 util / 1 other) | **none** — `expr2t::pretty` is an s-expr debug dump, not C syntax | Long-horizon: write IREP2-native C printer (Phase 2.7, own issue) |
| **String constants** | `string_constant.{cpp,h}`, `string2array.{cpp,h}`, `array2string.{cpp,h}` | **MIGRATE_CALLERS_FIRST** | `string2array`: **1** (`c_typecast.cpp:750`); `array2string`: **1** (`io.cpp:233`) | `constant_string2t` (has `to_array`/`at`); **lacks `mb_value()`** | Port `mb_value` onto `constant_string2t`; rewrite `get_string_constant` on `expr2tc` |
| **Legacy std subclasses** | `std_expr.h`, `std_types.{cpp,h}`, `std_code.{cpp,h}` | **MIGRATE_CALLERS_FIRST** | **161** external includers (79 front / 24 goto / 28 core* / 24 util / 6 other) | full `*2t` family in `irep2_{type,expr}.h` | Remove the **7** core migrate-seam round-trips; bulk blocked on frontends |
| **Misc symbol helpers** | `replace_symbol.{cpp,h}`, `fix_symbol.{cpp,h}`, `symbolic_types.*`, `python_types.*`, `namespace.*` | **MIGRATE_CALLERS_FIRST** | `replace_symbolt`: 1 non-util (`goto_convert.cpp:794`); `fix_symbolt`: 1 (`c_link.cpp`) | `namespacet` already has `follow(type2tc)`; no IREP2 symbol-substitution | Pre-seam legacy; retain until goto-convert/c-link move |
| **IREP2-native leaves** | `goto_expr_factory.*`, `fallible_calls.*`, `format_constant.*`, `array_name.*`, `type_byte_size.*` | **ALREADY_IREP2** | — | self | None (drop one dead `array_name` include) |
| **IR core + serialization** | `irep.{cpp,h}`, `irep_serialization.*`, `symbol_serialization.*`, `xml_irep.*` | **RETAIN_BOUNDARY** | on-disk + witness contract | none, by design | Never migrate; prove unchanged |
| **Migration seam** | `migrate.{cpp,h}` | **RETAIN_BOUNDARY** | `migrate_expr_back`: **109/109** kinds; `migrate_type_back`: **15/15** | — | Keep total; success = fewer *call sites*, not fewer arms |

\* Of the 28 `std_*` core includers, only **7** `.cpp` files genuinely
construct a legacy subclass (`memory_alloc`, `reachability_tree`,
`symex_main`, `value_set`, `value_set_analysis`, `symex_catch`,
`smt_conv`), and all use it as transient scratch at the migrate seam.

### Two corrections worth flagging (adversarial pass)

- The frontends are **not** entirely upstream of the migrate seam:
  `clang_c_adjust_expr.cpp:1311-1318` itself calls `migrate_type` +
  `migrate_expr_back` to synthesize a nondet side-effect. The seam is
  therefore *not* monotonically shrinkable to zero by util work alone.
- `irep2/irep2.h:44` directly `#include`s `util/irep.h`, and
  `irep2/irep2_type.h:6` includes `util/type.h` (the latter **vestigial**
  — `typet` is referenced 0× across `src/irep2/`). So legacy `irept`
  is a *compile-time* dependency of the IREP2 headers via the migration
  layer; `irep.{cpp,h}` cannot be deleted while `migrate.{h,cpp}` lives.

## 5. IREP2 equivalents and API gaps

The target surface is largely already present: `c_types.h` carries a
`*_type2`/`get_*_type` sibling for every C-type builder;
`irep2_type.h` macro-generates the `*_type2tc` factories;
`irep2_expr.h` has the typed constant factories
(`constant_int2tc`, `constant_bool2tc`, `constant_array_of2tc`, …);
`irep2_utils.h` has `gen_zero`/`gen_one`/`gen_true_expr`/`gen_false_expr`/
`make_not`/`conjunction`/`disjunction`/`gen_nondet` and a rich
`is_constant*` family.

**Confirmed gaps that must be built before specific callers can move:**

| Gap | Needed by | Effort |
|-----|-----------|--------|
| `to_integer(expr2tc, BigInt&)` — guarded fold-or-fail (constant, typecast-of-constant) | legacy-simplifier callers, `str_conv` | small; new helper |
| `from_double(double, const type2tc&)` → `constant_floatbv2tc` | `arith_tools` legacy retirement | trivial port of existing legacy body |
| `mb_value()` / `convert_mb` / `convert_utf8` on `constant_string2t` | `get_string_constant` (Phase 2.6) | medium; carry decoder over **verbatim** (endianness/surrogate hazards) |
| `expr_has_floatbv(expr2tc)` | misc | one-liner from existing predicates |
| value-returning `negate(const expr2tc&)` (wraps in-place `make_not`) | callers without an lvalue | trivial |
| `symbol2tc`-from-`symbolt` convenience | the ~60 `symbol_expr(symbolt)` sites | trivial; smooths migration |
| 13 missing `*_type2` C-type siblings (`enum`, `wchar`, `char16/32`, `half_float`, `uint256`, short/char variants, …) | goto/symex legacy `c_types` callers | small each |

**Intentional non-gaps** (no IREP2 analogue *by design*): `gen_unary`/
`gen_binary` (string-id generic builders — IREP2 is statically typed;
rewrite to typed factories per site), `make_binary` (n-ary rebalance —
IREP2 nodes are already binary), `make_next_state` (IREP2 models
renaming via `symbol2t` `rlevel`/`level*_num` fields). These have **zero
external callers** today and are dead-code retirements, not ports.

## 6. Phased, commit-sized decomposition

Ordered lowest-risk / unblocking-first. Each numbered item is one
reviewable commit; each phase is independently shippable. **Apply and
test one commit at a time** (incremental patch testing); do not batch.

### Phase 2.0 — Baseline & harness (no behaviour change)
1. Snapshot `migrate_census.sh src/util` into the tree as the `util`
   scoreboard; record the golden `capture_goto_baseline.sh` over a
   representative corpus (`regression/esbmc`, `regression/python`,
   `regression/esbmc-cpp`, `regression/floats`).
2. Add the **missing DUAL_API equivalence-test harness** (validation
   gap, §8): a unit pattern asserting `legacy_overload(migrate_back(x))
   == irep2_overload(x)`, modelled on `base_type.test.cpp`.

### Phase 2.1 — Dead-include & dead-code hygiene (RETIRE_DEAD)
3. Drop stale includes (no symbol used): `c_expr2string.h` in
   `document_subgoals.cpp`; `c_typecast.h` + `array_name.h` in
   `dereference.cpp`; `string_constant.h`/`string2array.h` in
   `pytest.cpp` + `execution_state.cpp`; `xml_irep.h`/`xml.h` in
   `show_claims.cpp` + `loop_numbers.cpp`; `simplify_expr.h` in
   `padding.cpp`, `clang-c-frontend/typecast.cpp`,
   `solidity-frontend/typecast.cpp`.
4. Delete dead `expr_util` helpers `gen_implies`, `make_next_state`,
   `make_binary`; inline `gen_unary` into its sole caller `gen_not`.
5. Delete dead `type_eq.{cpp,h}` and its CMake entry.
6. Drop the zero-caller legacy `base_type_eq(exprt,exprt)`,
   `base_type(exprt&)`, and matching `base_type_eqt` members.

*Each deletion is a branch removal → C-Dead Mode C proof (or cited
zero-caller grep as implicit discharge) + `diff_goto_baseline` exit 0.*

### Phase 2.2 — Retire the legacy CBMC simplifier (RETIRE_DEAD)
7. Redirect `migrate.cpp:153,172` to migrate-then-`simplify(expr2tc&)`.
8. Redirect `builtin_functions.cpp:119,354,409` likewise.
9. Redirect `str_conv.cpp:557` — **gated** on the per-rule parity
   check (§9 open question): its downstream branches on whether the
   result folded to a constant.
10. Redirect `c_typecast.cpp:919` `simplify_typecast` →
    `typecast2t::do_simplify`, preserving the `#c_sizeof_type` attribute.
11. Delete `simplify_expr.{cpp,h}`, `simplify_expr_class.h`,
    `simplify_utils.{cpp,h}`; drop from `CMakeLists.txt`.

### Phase 2.3 — Class-B migrate round-trip elimination (perf, util-internal)
12. `expr_simplifier.cpp`: replace the **9** `migrate_type_back`
    round-trips with `fixedbv_spect(fixedbv_type2t)`,
    `ieee_float_spect(floatbv_type2t)`, `from_integer(BigInt,type2tc)`
    (and build the 1 missing helper for the residual site). Must produce
    **bit-identical** float/fixedbv specs — high blast radius.
13. `namespace`: add a native `follow(type2tc)` that avoids the
    `migrate_type_back → follow → migrate_type` detour (hot path).
14. `c_sizeof`: retire the legacy `typet` overload **iff** its last
    legacy caller is gone (open question §9).

### Phase 2.4 — Core migrate-seam round-trip elimination (perf + surface)
15. `value_set::do_function_call` / `value_set_analysist`: compute on
    `code_type2t.arguments` / `struct_union_members` directly.
16. `memory_alloc.cpp`: build the dynamic type as `array_type2t`
    directly instead of staging `array_typet` + `constant_exprt`.
17. `smt_conv.cpp`: build fixedbv/floatbv constants from `type2tc`
    width/value (needs IREP2-native `fixedbvt`/`ieee_floatt` builders).
18. `symex_catch.cpp`: replace `to_struct_type(get_type())` base-class
    walk — **gated** on whether the `bases` sub-irep survives
    `migrate_type` (open question §9); if it does not, this site stays.
19. Prune now-incidental `std_*` includes from `irep2/*`.

### Phase 2.5 — DUAL_API_THEN_DROP: drop legacy overloads
20. Migrate `goto2c/expr2c.cpp:314` free `base_type(typet&)` to the
    `type2tc` overload (or keep `typet` locally and convert).
21. Migrate `c_link.cpp` (3 sites) and `c_typecast.cpp:691` from
    `symbolt::get_type()`/`typet` to `get_type2()`/`type2tc` — re-verify
    incomplete-type and link-merge behaviour (the two `base_type_eq`
    paths differ on `incomplete_struct`/`incomplete_array`/`subtype`).
22. Add the 13 missing `*_type2` C-type siblings; migrate goto/symex
    legacy `size_type`/`bool_type`/`char_type` sites onto them.
23. Remove the vestigial `util/type.h` include from `irep2_type.h`.
24. After the above + frontends, drop the remaining legacy `typet`
    `base_type_eq` / `c_typecast` / `arith_tools` overloads.

### Phase 2.6 — String-constant decoder un-trapping
25. Port `mb_value`/`convert_mb`/`convert_utf8` onto `constant_string2t`
    (legacy `mb_value` delegates during transition).
26. Rewrite `get_string_constant` (`builtin_functions.cpp`) on `expr2tc`.
27. Inline `string2array` into `c_typecast`'s sole call site (already a
    thin IREP2 wrapper); give `array2string` an IREP2 form or build
    `constant_string2t` directly in `io.cpp`.

### Phase 2.7 — IREP2-native C/C++ pretty-printer (long-horizon, own issue)
28. New `expr2tc`/`type2tc` C-syntax walker behind the existing
    `from_expr(expr2tc)`/`from_type(type2tc)` overloads
    (`langapi/language_util.h`), initially delegating to the legacy path
    for golden-output baselining.
29. Port `c_expr2string`'s ~159 cases case-by-case, gated by
    golden-output regression diffing against the legacy printer.
30. Re-point the IREP2 overloads at the native printer, removing the
    `migrate_*_back` hop; keep legacy `c_expr2string` for the frontend
    `from_expr(exprt)` path and `goto2c` (compilable-C emitter).

**Phase 2.7 is high-risk and explicitly its own tracking issue** —
counterexample/witness text is user- and CI-visible and matched by
`test.desc` regexes; SV-COMP/GraphML witness parity must hold.

## 7. Risk register

| # | Area | Risk | Severity | Mitigation |
|---|------|------|----------|------------|
| R1 | correctness | Redirecting a caller from legacy `simplify(exprt&)` to `simplify(expr2tc&)` assumes per-rule parity; `str_conv` branches on whether the result became constant. | high | Per-rule diff before Phase 2.2 commit 9; equivalence unit test. |
| R2 | correctness | The two `base_type_eq` overloads differ on `incomplete_struct`/`incomplete_array`/`subtype` ids; `is_subclass_of(type2tc)` deliberately **transposes** args (`base_type.cpp:434-442`) and `dereference`/`smt_casts` rely on the inversion. | high | DUAL_API equivalence test; preserve the inversion verbatim; per-caller re-verify. |
| R3 | correctness | `from_integer(BigInt,type2tc)` truncates out-of-range values via a binary round-trip; the legacy `typet` overload stores without explicit wrap. | med | Confirm no legacy caller depends on un-truncated behaviour before swapping. |
| R4 | correctness | Class-B float/fixedbv spec rebuild (Phase 2.3) must be **bit-identical**; the simplifier touches every solver path. | med (high blast radius) | Constant-fold equivalence test; dual-solver regression. |
| R5 | correctness | Core seam round-trips (Phase 2.4) rely on `get_type()`'s lazy cache; the `bases` sub-irep walked in `symex_catch` may have **no** IREP2 representation. | med | Gate commit 18 on a `migrate_type` faithfulness check for `bases`. |
| R6 | serialization | `irep_serialization`/`symbol_serialization` define the on-disk format; any `irept`-layout / `full_hash` / S·N·C-tag change breaks old binaries. | high | RETAIN_BOUNDARY; goto-binary round-trip regression; **never edit**. |
| R7 | compatibility | A new IREP2 C printer (2.7) must reproduce `c_expr2string`'s exact text; witness/CI output is regex-matched. | high | Golden-output diff gate per case; keep legacy printer for frontends. |
| R8 | compatibility | `irep2.h` depends on `util/irep.h`; `exprt`/`typet`/`codet`/`locationt` derive from `irept`. The legacy core cannot be deleted while the seam or frontends exist. | high | Scope excludes the IR core; documented as RETAIN_BOUNDARY. |
| R9 | correctness | IREP2 `c_typecast`/`follow_with_qualifiers` **drops** C qualifiers (`const`/`volatile`/`restrict`) (TODO `c_typecast.cpp:300`); migrating frontends onto it silently loses const-correctness diagnostics. | med | Restore qualifier handling before any frontend redirect; out of util scope. |
| R10 | correctness | `mb_value` (multibyte/UTF-8/wide, endianness, locale) is the trickiest logic in the tree and lives only on the legacy class. | med | Carry the decoder over **verbatim**; property-test against CPython where the Python frontend exercises it. |
| R11 | performance | Reordering "simplify-then-migrate" to "migrate-then-simplify" (2.2) and seam removals (2.3/2.4) change when allocation cost is paid on hot goto-convert/symex paths. | low (net win) | Census + spot benchmark; the seam removals are net performance *gains*. |
| R12 | correctness | Name-identical legacy/IREP2 overloads (`gen_zero`, `to_struct_type`, `base_type_eq`) resolve by argument type; a mechanical migration can silently bind the wrong overload (both compile). | med | Per-site verify the resolved overload; lean on `-Werror` + unit equivalence tests. |

## 8. Validation, regression & equivalence-checking requirements

Reuse the established Part I gate verbatim; **per-tier additions**:

**Universal PR gate** (the `scripts/irep2-migration/` checklist):
1. `migrate_census.sh <target>` **strictly decreases** in the touched
   region, increases nowhere.
2. `diff_goto_baseline.sh <corpus>` exits **0** (canonical
   `--goto-functions-only` dump bit-for-bit identical after
   `irep2_canon`).
3. Regression verdicts **and counterexamples** agree under **both
   Bitwuzla and Z3** (dual-solver mandatory).
4. Affected unit suites green: `ctest -R migratetest|symboltest|
   base_typetest|simplify2ttest`; full unit run `ctest -LE regression`.
5. Run the affected regression corpus on an **asserts-enabled build** so
   the `NDEBUG` migrate cross-check (`migrate.cpp:380-424`) stays live —
   it is disabled under `NDEBUG`. For Python-touching changes also run
   `scripts/check_python_tests.sh`.

**Per tier:**
- **RETIRE_DEAD / drop step** — the C-Dead **Mode C** proof
  (`__ESBMC_unreachable()` + `--enable-unreachability-intrinsic`, dual
  solver) that the removed path is unreachable pre-patch, *or* cite an
  existing reproducer as implicit discharge. The census is necessary but
  **not sufficient** to prove deadness.
- **DUAL_API_THEN_DROP** — **before** dropping a legacy overload, add a
  unit equivalence test asserting `legacy(migrate_back(x)) == irep2(x)`
  over the kinds the overload handles (this harness does **not exist
  today** — Phase 2.0 builds it). Keep it as the durable Phase-2
  contract regression.
- **MIGRATE_CALLERS_FIRST** — gating only. The `NDEBUG` cross-check +
  `diff_goto_baseline` remain the equivalence guarantee that the legacy
  form flowing through the boundary is still faithfully convertible.
- **RETAIN_BOUNDARY** — inverse validation: prove it is *not* changed.
  Goto-binary serialise/deserialise round-trip + the full goto-binary
  regression suite; old binaries must still load.

**Known validation gaps** (must be closed as part of the plan, not
assumed):
- No API-level legacy-vs-IREP2 equivalence test exists for DUAL_API
  files — Phase 2.0 adds the pattern.
- `diff_goto_baseline` only catches drift **visible in the goto dump**;
  a change whose only effect is in later symex/solver encoding relies on
  full dual-solver verdict+counterexample agreement instead.
- `migrate_census` counts call sites, not dead paths — pair with the
  C-Dead proof on every deletion.

### 8.1 Environment caveat — non-deterministic operational-model goto

**`diff_goto_baseline` is only reliable *within a single build*, not
across rebuilds.** ESBMC's bundled C operational models
(`src/c2goto/library/*` → FLAIL-mangled `clib*.goto`) and Python models
(`src/python-frontend/models/*`) are regenerated on *every* build, and
that generation is **non-deterministic**: regenerating `clib*.goto` with
the *same* `esbmc` binary yields different SHA-256 digests
(empirically confirmed on the Phase 2.1 work, 2026-05-29). The
nondeterminism surfaces in the `--goto-functions-only` dump as
per-run-varying synthesized names — e.g. nested-function filenames
(`esbmc-nested.<hex>-<hex>.c`), unpack temporaries
(`ESBMC_unpack_temp_<address>`), and the `esbmc-python-astgen-<hex>`
temp-dir path encoded as a `__file__` char array.

Consequence: for *any* source change that triggers a rebuild (all of
them), `diff_goto_baseline` against a baseline captured on a different
build reports spurious diffs in every test whose dump surfaces an
affected library/model function — independent of whether the source
change altered behaviour. A "clean" diff can be luck (this build's model
bake happened to match the baseline's) and a "dirty" diff can be pure
build noise. The Phase 2.1 dead-code commits each showed this: build *N*
produced ~69 differing tests vs the baseline, build *N+1* (no source
change to the affected paths) produced ~2, and running one binary twice
reproduced every residual diff — proving the differences were model
nondeterminism, not the patch.

**Recommended gate for rebuild-triggering changes — deterministic
regression-verdict comparison.** Verdicts (and counterexample
reachability) are *unaffected* by model symbol-naming nondeterminism, so:
1. On a **clean baseline build** (e.g. `master`), capture the pass/fail
   set over a stratified `regression/{esbmc,floats,esbmc-cpp,python}`
   subset.
2. On the **post-change build**, confirm the pass/fail set is
   **identical** (same failures, which on a given host are the
   pre-existing platform/solver-specific ones).
A behaviour-preserving change leaves the set unchanged.

Use `diff_goto_baseline` only (a) *within one build* — run the same
binary twice; any difference is runtime nondeterminism, isolating it from
build-level model nondeterminism — or (b) after extending `irep2_canon`
(`scripts/irep2-migration/lib.sh`) to mask the nondeterministic loci
above. Note also that `irep2_canon`'s temp-dir rules cover
`/var/folders/.../T/esbmc…` and `/tmp/esbmc…` but **not** a
`$TMPDIR`-nested layout such as `/tmp/<dir>/esbmc.<hex>/headers/…`; add a
rule for the host's actual `$TMPDIR` shape before trusting a cross-build
diff there.

## 9. Assumptions, dependencies, and open design questions

**Assumptions** (each must hold or the affected phase pauses):
- The frontends remain a **permanent legacy boundary** (Part I, B1).
  Every MIGRATE_CALLERS_FIRST item is blocked on this not changing.
- The goto-binary on-disk format **does not change** (Part I, B4).
- The IREP2 simplifier is a behavioural **superset** of the legacy one
  for the 7 redirected sites — *unverified*; see Q1.

**Dependencies:**
- Phase 2.7 depends on a decision about witness-text parity (Q4) and on
  `goto2c` keeping a compilable-C emitter (it may need a distinct
  emitter, not the shared printer).
- Phase 2.4 commit 17 depends on IREP2-native `fixedbvt`/`ieee_floatt`
  builders that take `type2tc` + value without `constant_exprt`.
- Phase 2.5 depends on the frontends (for the *final* legacy-overload
  drop) and on the `c_typecast` qualifier TODO (R9).

**Open questions** (resolve before the cited commit):
- **Q1** — Is the IREP2 simplifier a strict superset of `simplify_exprt`
  for typecast-to-bool folding, redundant-typecast elimination,
  if-branch and alloc-size constant folding? (blocks 2.2 commit 9)
- **Q2** — Does legacy `simplify_typecast` (`simpl_const_objects=false`)
  perform normalization `typecast2t::do_simplify` does not? (blocks 2.2
  commit 10)
- **Q3** — Does the `bases` sub-irep walked in `symex_catch`
  (`st.find("bases")`) survive `migrate_type` into a faithful
  `struct_type2t`, or is base-class info legacy-only? (blocks 2.4
  commit 18)
- **Q4** — Does SV-COMP/GraphML witness generation regex-match the exact
  `c_expr2string` text, or only structural fields? (sets the parity bar
  for 2.7)
- **Q5** — Do legacy `gen_zero`'s `complex`/`c_enum` branches (the IREP2
  twin aborts on them) get exercised by any frontend before a redirect?
- **Q6** — Is the legacy `c_sizeof(typet)` overload dead, or
  frontend-only? (decides 2.3 commit 14)

## 10. Success metrics and exit criteria

- **Primary metric:** `migrate_census.sh src/util` total
  (`migrate_*` + `migrate_*_back` call sites) strictly decreasing
  PR-over-PR. Note the **arm count cannot shrink** — all 109 `expr2t` /
  15 `type2t` back-arms stay live, pinned by `symbolt`'s lazy legacy
  cache and the `migrate.test.cpp` totality invariant; the metric is
  *call sites*, not coverage.
- **Secondary:** the `util/` IREP2 share (the Part I surface-ratio
  census) rising from 35 %.
- **Exit (this plan):** every RETIRE_DEAD and DUAL_API_THEN_DROP item
  discharged; Class-A/B split documented in `migrate.h`; the residual
  legacy surface is exactly the **RETAIN_BOUNDARY** set (IR core,
  on-disk format, migrate seam) plus the **MIGRATE_CALLERS_FIRST** set
  pinned by the frontend boundary. Phase 2.7 (the C printer) and any
  frontend migration are explicitly **out of scope** and carry their own
  tracking issues.

The bar is Part I's bar: **behaviour-preserving at every step**, proven,
not assumed.

## 11. Outcome — Phase 2 concluded at the frontend boundary

> **Status: concluded.** Sections 1–10 above were the forward plan;
> this records what actually landed and where the focused follow-on
> stopped. The non-frontend `util/` → IREP2 surface is drained; the
> residual is exactly the **RETAIN_BOUNDARY** + **MIGRATE_CALLERS_FIRST**
> sets the plan anticipated, pinned by the permanent frontend boundary.

### What landed

| PR | Phase | Outcome |
|----|-------|---------|
| [#4935](https://github.com/esbmc/esbmc/pull/4935) | 2.0 + 2.1 | Census scoreboard (`scripts/irep2-migration/census-util.txt`); RETIRE_DEAD hygiene — dropped stale includes, dead `expr_util` helpers, `type_eq.{cpp,h}`, the zero-caller `exprt` `base_type`/`base_type_eq` overloads. |
| [#4938](https://github.com/esbmc/esbmc/pull/4938) | 2.2 | Retired the legacy CBMC simplifier (`simplify_expr.{cpp,h}`, `simplify_expr_class.h`, `simplify_utils.{cpp,h}`, ~2964 lines); built `to_integer(expr2tc)`; redirected all 7 caller sites to the IREP2 simplifier (Q1/Q2 resolved). |
| [#4941](https://github.com/esbmc/esbmc/pull/4941) | 2.3 | Replaced 8/9 `expr_simplifier` `migrate_type_back` round-trips with native `fixedbv_spect`/`ieee_float_spect`/`from_integer(type2tc)`; native `namespacet::follow(type2tc)`. |
| [#4944](https://github.com/esbmc/esbmc/pull/4944) | 2.4 | Core migrate-seam round-trips removed: `value_set::do_function_call` reads `get_type2()`; `smt_conv::get_by_value` builds model fixedbv/floatbv natively; pruned stale `std_*` includes from `irep2/`. |
| [#4946](https://github.com/esbmc/esbmc/pull/4946) | post-2.4 | Eliminated further migrate-back sites across goto-programs / pointer-analysis (goto_check, goto_inline, dereference, goto_convert_functions, …). |
| [#4947](https://github.com/esbmc/esbmc/pull/4947) | 2.5 | Dropped the vestigial `util/type.h` include from `irep2_type.h`; documented the frontend boundary as reached. |

### Resolved open questions and per-commit blockers

The plan's gated items resolved against the tree as follows; these stay
on the legacy side because the IREP2 side genuinely cannot represent them
or the callers are frontend-pinned:

- **Q1 / Q2** (simplifier parity, Phase 2.2) — **resolved, redirected.**
  `typecast2t::do_simplify` folds typecast-of-constant identically; the
  one live `simplify_typecast` site is guarded by `op0().is_constant()`,
  so only constant folding matters.
- **Q3** (`bases` sub-irep, Phase 2.4 commit 18) — **resolved: stays.**
  `struct_type2t` has no `bases` field and `migrate_type` never copies
  it; C++ base-class metadata is legacy-only, so `symex_catch`'s
  base-class walk keeps reading the legacy `get_type().find("bases")`.
- **Q5** (`gen_zero` complex, Phase 2.3 residual) — `gen_zero(type2tc)`
  `abort()`s on `complex`, but the complex-to-bool cast does not reach
  the `typecast2t::do_simplify` bool branch (the clang frontend lowers it
  to part-wise comparisons first), so [#4946](https://github.com/esbmc/esbmc/pull/4946)
  safely migrated the site. The `gen_zero(type2tc)` complex gap remains a
  latent defensive hole, not an active bug.
- **Q6** (legacy `c_sizeof(typet)`, Phase 2.5 commit 14) — **frontend-only,
  not dead.** Its 3 callers are 2 frontends + `goto2c`; cannot drop.
- **Phase 2.4 commit 16** (`memory_alloc` → `array_type2t` directly) —
  **blocked.** `array_type2t` has no `dynamic`/`alignment` field, and
  `symex_valid_object` reads `symbol.get_type().dynamic()` for
  dynamic-memory valid-object checks; building the type directly would
  drop that flag. Kept the legacy `array_typet` construction.
- **Phase 2.5 commits 20/21/22/24** — **frontend-gated / risky / premature.**
  The DUAL_API overload drop (24) is "after frontends"; the `c_link`
  `base_type_eq` migration (21) is risky (R2 — `typet`/`type2tc` paths
  differ on incomplete-struct/array/subtype, and `c_link` link-merges
  incomplete types); the missing `*_type2` siblings (22) have only
  frontend callers.
- **Phase 2.6** (string-constant decoder) — **pinned.** `mb_value`'s only
  callers are legacy goto-convert (`get_string_constant`) and the python
  frontend (`python_set`); `get_string_constant`'s callers pass legacy
  `exprt::operandst`. The one non-frontend opportunity (`array2string` at
  `io.cpp`) is a single small printf-`%s` round-trip, left for a future
  focused change.
- **Phase 2.7** (IREP2-native C printer) — out of scope, its own issue.

### Methodology note

The differential goto-baseline harness (`scripts/irep2-migration/`) proved
**unreliable across rebuilds** in some environments: ESBMC re-bakes its
bundled operational-model goto non-deterministically per build (§8.1).
Every Phase 2 PR was therefore gated on **deterministic regression-verdict
equivalence against clean `master`** over a stratified corpus, plus the
affected unit suites and per-change targeted oracles — not on the goto
diff. See §8.1 for the full rationale.

### Surface ratio at the Phase 2 close (`src/util`)

Measured by IREP2- vs legacy-IR type-name **line mentions** over `src/util`
(`git grep -P '\b[A-Za-z_]+2tc?\b'` for the `*2t`/`*2tc` family vs
`git grep -P '\b([A-Za-z_]*(exprt|typet|codet)|irept)\b'` for the legacy
family):

| `src/util` | IREP2 lines | legacy lines | IREP2 share |
|------------|------------:|-------------:|------------:|
| Part II baseline (`6c4d610694`) | 2845 | 2584 | **52 %** |
| Phase 2 close (post-#4947)      | 2858 | 2301 | **55 %** |

So legacy IR type-name mentions in `util/` fell by ~283 lines (the IREP2
share rising 52 % → 55 %). The bigger story is the **whole legacy units
deleted**: `simplify_expr.{cpp,h}` + `simplify_expr_class.h` +
`simplify_utils.{cpp,h}` (~2964 lines), `type_eq.{cpp,h}`, the dead
`expr_util` helpers, and the zero-caller `exprt` `base_type` overloads.
The residual legacy in `util/` is the frontend-pinned helper layer
(`c_types`/`c_typecast`/`std_*` builders, `c_expr2string`/`type2name`),
the `migrate_*` seam, and the IR core — i.e. exactly the RETAIN_BOUNDARY +
MIGRATE_CALLERS_FIRST sets.

*(This line-mention metric is reproducible but distinct from — and not
directly comparable to — Part I's narrower per-type "Surface ratio at the
close" table, which reported `util/` at 35 %.)*

### Bottom line

The clean, non-frontend `util/` → IREP2 migration is **complete**. The
remaining legacy surface is the deliberately-retained boundary: the IR
core and on-disk goto-binary format (`RETAIN_BOUNDARY`), the `migrate_*`
seam (kept with its cross-check), and the frontend-pinned helper layer
(`MIGRATE_CALLERS_FIRST`). Further progress requires migrating the
frontends themselves — the permanent boundary Part I declared out of
scope — and so belongs to its own, much larger tracking effort.

---

# Part III — Solidity frontend → IREP2 (forward plan)

> **Status: planning.** Parts I and II are closed records. This part is a
> *forward plan* that deliberately **reopens boundary B1** ("Frontend →
> goto input", marked *Deferred indefinitely* in Part I) for **one
> frontend only — Solidity**. Nothing here has landed. Tracking issue:
> **to be filed.** It must not be opened under the umbrella issue that
> closed with the Part I retrospective; B1 was declared out of scope
> there, and this plan changes that scope deliberately and with evidence.

This part is written to be executable by an engineer who has *not* worked
on the Solidity frontend before. Every load-bearing claim carries a
`file:line` anchor and a re-verification command so you can confirm the
tree has not drifted since this was written.

## 1. Scope, motivation, and the non-negotiable constraint

`src/solidity-frontend/` is ~23.5 kLOC across 27 files. Its docstrings
already *claim* it is an "AST-to-irep2 converter" (`solidity_convert.h:1-9`)
— that is aspirational, not factual: the converter is **100 % legacy
IR** today (`exprt`/`typet`/`codet`/`symbolt`). This plan closes the gap
between the docstring and reality, **for the parts that can be closed
soundly**, and documents precisely which parts cannot move without
prerequisite infrastructure shared with the clang frontends.

Why Solidity is the right pilot for reopening B1:

- It is the most self-contained pure-legacy frontend. Unlike clang-c /
  clang-cpp it has no Clang AST objects flowing through it (it consumes a
  solc JSON AST), and unlike the python frontend it does not also drive a
  large operational-model surface from the same converter.
- Smart-contract verification is a soundness-critical domain where the
  compile-time-typing benefit of IREP2 (Part I §"Why this mattered") pays
  off directly: a misclassified expression (mapping vs array, `bytesN`
  width, storage vs memory aliasing) is a *silent verdict corruption*,
  not a crash.
- It carries the largest concentration of **Solidity-specific irep
  string attributes** (§4, F4) — the exact construct IREP2 was designed
  to make impossible. De-risking that surface is valuable on its own,
  independent of how far the migration ultimately goes.

**The constraint is Part I's constraint, restated and non-negotiable:**
ESBMC is a verifier; **every step must be behaviour-preserving** —
identical pass/fail verdicts, identical counterexample-visible text where
`test.desc` matches it, dual-solver agreement, no on-disk goto-binary
format change. A step that is merely "more IREP2" but shifts one
Solidity verdict is a regression, not progress. Where this plan and
implementation convenience conflict, correctness wins — the user has
made this explicit and it is the governing rule of this part.

## 2. How this plan was derived (reproducibility)

Same method as Part II §2, re-run against the Solidity tree:

- **Idiom census** — `grep` over `src/solidity-frontend/**` for legacy
  builders (`symbol_expr`, `side_effect_expr_function_callt`,
  `code_*t`, `member_exprt`, …), for direct irep surgery
  (`move_to_operands`, `copy_to_operands`, `.operands()`, `.find(`,
  `.id()`), and for every `.set("#…")` / `.get("#…")` attribute key.
- **Seam trace** — read `symbol.{h,cpp}`, `migrate.{h,cpp}`,
  `solidity_language.cpp`, `clang_c_adjust_expr.cpp`, `c_link.cpp`,
  `goto_convert_functions.cpp` to fix exactly where legacy becomes IREP2.
- **IREP2 API audit** — read `irep2_expr.h`, `irep2_type.h`,
  `expr_kinds.inc`, `type_kinds.inc`, `c_types.h` for the available
  factories and the representational gaps.
- **Adversarial re-check** — every load-bearing fact in §4 was
  re-derived by an independent pass instructed to refute it; the figures
  below are the post-correction ones.

Re-run before acting on any phase; counts are a snapshot.

## 3. Where the legacy→IREP2 boundary sits today (the seam)

The symbol table is the pivot (Part I, B2). `symbolt` stores
`type2tc`/`expr2tc` as the source of truth with **lazy legacy caches**
(`symbol.h:89-99`). The two setters behave asymmetrically:

- `set_type(const typet&)` / `set_value(const exprt&)` store the **legacy**
  form and invalidate the IREP2 side (`symbol.cpp:36-41`). *No eager
  forward-migration* (Part I, S5a hotfix rationale).
- `set_type(const type2tc&)` / `set_value(const expr2tc&)` store **IREP2**
  and invalidate the legacy side.

The Solidity converter exclusively uses the **legacy** setters via
`get_default_symbol` + `move_symbol_to_context`
(`solidity_convert_util.cpp:200`; it never touches `get_type2()` /
`get_value2()` / `set_type(type2tc)`). So a Solidity symbol enters the
table legacy-valid / IREP2-invalid.

After the converter runs, **two shared legacy passes run over its output**
inside `solidity_languaget::typecheck` (`solidity_language.cpp:333-387`):

1. `clang_cpp_adjust adjuster(new_context); adjuster.adjust();`
   (`solidity_language.cpp:370`). Adjust reads `get_value()`/`get_type()`
   (legacy form, lazily back-migrated if needed), mutates **in place as
   legacy**, and writes the legacy form back
   (`clang_c_adjust_expr.cpp:56-75`). It does not consult IREP2.
2. `c_link(context, new_context, module)` (`solidity_language.cpp:382`).
   Link compares and merges symbols using `base_type_eq` on the **legacy
   `typet`** and reads `get_value()` for body swaps (`c_link.cpp`).

Only later, in `goto_convert`, does legacy become IREP2 for good:
`convert_function` reads the body as **legacy `codet`**
(`to_code(symbol.get_value())`, `goto_convert_functions.cpp:116`) and
`goto_convert_rec` lowers it (`:139`); `goto_convertt::copy` calls
`migrate_expr(code, t->code)` per instruction. The function *type* goes
through the `migrate_symbol_type` chokepoint (`:104`).

```
solc JSON ──► solidity_convertert ──► symbolt (LEGACY valid)
                                          │
                       clang_cpp_adjust  ─┤  reads/writes LEGACY in place
                              c_link     ─┘  compares LEGACY typet
                                          ▼
                       goto_convert: to_code(get_value())  ← LEGACY codet
                       goto_convert_rec lowers structured CF
                       migrate_expr per-instruction        ──► IREP2 (permanent)
```

## 4. Architectural findings that bound this plan

Each is verified; the re-check command is given.

**F1 — Symbol table is IREP2-source-of-truth with lazy legacy caches.**
`symbol.h:89-99`, `symbol.cpp:36-55,82-141`. A frontend *may* legally
call `set_type(type2tc)` / `set_value(expr2tc)`. Re-check:
`sed -n '60,160p' src/util/symbol.cpp`.

**F2 — Function bodies are hard-pinned to legacy `codet`; IREP2 has no
structured control flow.** `goto_convert_functions.cpp:110-139` reads the
body as `codet` and `goto_convert_rec` lowers it. `expr_kinds.inc`
contains only the **lowered, goto-level** code kinds — `code_block`,
`code_assign`, `code_decl`, `code_dead`, `code_expression`,
`code_return`, `code_skip`, `code_goto`, `code_function_call`,
`code_printf`, `code_comma`, `code_asm`, and the `code_cpp_*` exception
forms. There is **no** `code_ifthenelse2t`, `code_while2t`,
`code_for2t`, `code_switch2t`, `code_dowhile2t`, `code_break2t`,
`code_continue2t`, or `code_label2t`. The Solidity emitter currently
produces all of these structured forms (`code_whilet` ×2,
`code_switcht`, `code_fort`, `code_dowhilet`, `code_breakt` ×2,
`code_continuet`, `code_labelt` ×19, plus `if`). **Lowering structured
control flow to goto is `goto_convert`'s job and lives below the frontend
boundary.** Re-check: `grep -E 'IREP2_EXPR\(code_' src/irep2/expr_kinds.inc`
and `grep -roE 'code_[a-z]+t' src/solidity-frontend | sort -u`.

→ **Consequence (PIN P1):** the frontend cannot emit IREP2 *function
bodies* and feed them to `goto_convert`. Bodies remain legacy `codet`
until an IREP2-native goto-convert input language exists — a separate,
much larger project, explicitly out of scope here.

**F3 — Two shared legacy passes post-process Solidity output.**
`clang_cpp_adjust` and `c_link` (§3) are shared verbatim with the clang
frontends and operate on legacy IR. Even if the converter stored IREP2,
adjust would read it back as legacy (`migrate_*_back`), mutate legacy,
and re-store legacy — round-tripping it immediately, **and dropping any
Solidity attribute on the way (see F4).** Re-check:
`grep -n 'clang_cpp_adjust\|c_link' src/solidity-frontend/solidity_language.cpp`.

→ **Consequence (PIN P2):** "emit IREP2 directly from the converter"
delivers nothing — and is actively *unsound* w.r.t. F4 — while adjust/link
remain legacy passes. Migrating them is shared infrastructure work, out
of scope here.

**F4 — `migrate_type`/`migrate_expr` silently drop every Solidity irep
attribute. THIS IS THE CENTRAL SOUNDNESS HAZARD.** `migrate_type`
dispatches on `type.id()` and builds a fixed-field `type2tc`; the struct
arm (`migrate.cpp:191-214`) copies only `members`, `names` (`a_name`),
`pretty_names` (`a_pretty_name`), `tag`/`name`, and `packed`. There is no
code path anywhere in `migrate.cpp` that copies a `#`-prefixed attribute,
because `type2t`/`expr2t` have **no generic string-attribute map** — their
fields are fixed at construction (`irep2.h`, the `fields` tuple per kind).
The Solidity frontend attaches ≥20 distinct semantic attributes to
`typet`/`exprt`:

| Attribute | Carries | Set / Read | Migrates? |
|---|---|---|---|
| `#sol_type` | Solidity type class (CONTRACT, MAPPING, DYNARRAY, BYTES, …) | 1 set / **~43 read** | **dropped** |
| `#sol_array_size` | array / dynarray length | 8 / 13 | maps → `array_type2t.array_size` |
| `#sol_bytesn_size` | `bytes1`..`bytes32` width | 5 / 11 | **dropped** (no field) |
| `#sol_contract` | contract name on contract types | 1 / 6 | **dropped** |
| `#sol_data_loc` | `memory` / `storage` / `calldata` | 3 / 0 | **dropped** |
| `#sol_state_var` | state-variable flag | 2 / 1 | **dropped** |
| `#sol_name`, `#sol_mapping_array`, `#sol_dynarray_state`, `#sol_tuple_id`, `#sol_unchecked`, `#is_sol_virtual`, `#is_sol_override`, `#inlined`, `#member_name`, `#cpp_type`/`cpp_type`, `#zero_initializer`, `#is_modifier_placeholder`, `#c_sizeof_type`, `#cformat` | misc semantics & markers | various | **dropped** |

Today this is invisible **only because** the converter writes legacy into
the symbol's legacy cache and every `get_sol_type` / `.get("#…")` read is
served from that same un-round-tripped legacy `typet`. The instant a
Solidity type or expression crosses the IREP2 boundary (a `set_type(type2tc)`,
a `migrate_type`, or a symbol whose IREP2 side becomes source-of-truth),
the Solidity semantics **vanish without a diagnostic**. Re-check:
`grep -rn 'set("#sol\|get("#sol\|set_sol_type\|get_sol_type' src/solidity-frontend | wc -l`
and confirm `migrate.cpp` has no `#sol` handling:
`grep -n '#sol\|#member_name\|#is_sol' src/util/migrate.cpp` (returns nothing).

→ **Consequence:** the *first and most important* phase is to remove the
frontend's dependence on IR-carried Solidity semantics (§7, Phase 3.1),
**before** any node migrates. This is a soundness hardening even if no
further migration ever happens.

**F5 — The IREP2 construction API is largely present; the gaps are
enumerable.** Factories exist for every type/expr the frontend needs as
typed builders (`irep2_type.h:16-506`, `irep2_expr.h:104-1900`,
`c_types.h:65-79`). The confirmed gaps are listed in §6.

## 5. Target end-state: "IREP2-internal, legacy-at-the-seam"

Given P1+P2+F4, the **achievable, sound** target for frontend-scoped work
is *not* "the converter emits IREP2 end-to-end." It is:

> The converter computes and carries Solidity semantics in a **typed,
> off-IR companion structure** (never in irep string attributes), and
> constructs **types and the expression operands of statements** using
> IREP2 typed factories. Structured control-flow statements and the
> function-body `codet` handed to `goto_convert` remain legacy by design
> (P1). The single legacy hand-off is localized and documented.

```
                       ┌─────────────────── frontend scope ──────────────────┐
solc JSON ──► classify (typed sol_typeinfo, off-IR)
                  │                                            PIN P1: bodies
                  ▼                                            stay legacy codet
            build type2tc / expr2tc operands  ──► lower to legacy codet body
                                                    at ONE hand-off (migrate_back)
                       └──────────────────────────────────────────────────────┘
                                                    │
                            clang_cpp_adjust, c_link │ (PIN P2: shared legacy)
                                                    ▼
                                         goto_convert ──► IREP2 (permanent)
```

This buys the IREP2 compile-time-typing soundness benefit *inside the
converter* (a mis-built node fails to compile, not at symex), eliminates
the attribute hazard (F4) entirely, and leaves the externally-observable
behaviour bit-identical because the bytes that reach `clang_cpp_adjust`
are the same legacy `codet`/`typet` as today — just produced through a
typed path. The deeper boundary (P1/P2) is then a *clean, single, audited
seam* rather than a diffuse one, which is the precondition any future
"push the boundary below adjust/link/goto-convert" project would need.

### 5.1 The `sol_typeinfo` companion (replaces all `#sol_*` on IR)

Introduce one typed, frontend-local value type (proposed
`src/solidity-frontend/sol_typeinfo.h`):

```cpp
struct sol_typeinfo {
  SolidityGrammar::SolType cls = SolType::SolTypeError; // was #sol_type
  std::string contract_name;            // was #sol_contract
  data_location loc = data_location::none; // was #sol_data_loc
  unsigned bytesn_width = 0;            // was #sol_bytesn_size (bytes)
  bool is_state_var = false;           // was #sol_state_var
  bool is_mapping_backed_array = false;// was #sol_mapping_array
  // … one typed field per surviving attribute …
};
```

Carry it **keyed by a stable identity that does not depend on the IR
node surviving migration**:

- For symbols: a side-table `std::unordered_map<irep_idt, sol_typeinfo>`
  keyed by the symbol `id`. The python frontend already uses this
  pattern (`std::unordered_map<std::string, const symbolt*>` caches, e.g.
  `python-frontend/function_call/expr.h:437`).
- For transient sub-expressions during a single conversion: thread the
  `sol_typeinfo` as an explicit out-parameter / local alongside the
  `typet` being built, mirroring how `get_type_description(json, typet&)`
  already threads the JSON node.

The design rule that makes this *sound*: **classification must be
derivable from the solc JSON (or the companion), never read back off an
irep node.** Once that holds, the IR types are "plain" C-like types that
`migrate_type` converts losslessly — there is nothing left to drop.

> **Rejected alternative — extend `type2t`/`expr2t` with a generic
> attribute map.** This would re-introduce exactly the string-keyed,
> parse-at-every-access escape hatch IREP2 was built to abolish (Part I
> §"Why this mattered"), forfeiting the determinism and compile-time
> soundness that justify the whole migration. Do not do this. The
> companion structure is typed and frontend-local; it never enters the
> verifier core.

## 6. IREP2 API gaps to close before migrating construction

From the F5 audit. Build these (with unit round-trip tests) **before**
the phase that needs them; each is small.

| Gap | Needed by | Disposition |
|---|---|---|
| No `type2t`/`expr2t` attribute map | all `#sol_*` reads | **By design** — solved by §5.1 companion, not by extending IREP2. |
| `bytesN` width has no type field | `#sol_bytesn_size` consumers | Carry width in `sol_typeinfo`; represent the value as `unsignedbv_type2t(8*N)` or the existing `bytesN` operational-model struct — **decide once** (Q3) and apply uniformly. |
| Structured control-flow code kinds | function bodies | **Out of scope (P1).** Bodies stay legacy `codet`. |
| Expression-context function call | the 195 `side_effect_expr_function_callt` sites | Use `sideeffect2t(..., sideeffect_allockind::function_call)` (`irep2_expr.h:1743-1760`); statement-level uses `code_function_call2t` (`:1891-1900`). |
| `string_constant::mb_value` | string/bytes literals | `constant_string2t` lacks `mb_value` (Part II Phase 2.6, R10); port the decoder **verbatim** or keep string-literal lowering legacy at the seam. Gated by Q4. |
| 256-bit integer types | every Solidity `uintN`/`intN` | `unsignedbv_type2tc(256)` / `signedbv_type2tc(256)` already work; no new factory needed. Verify BigInt/SMT width handling under overflow checks (R-S5). |
| `symbol2tc` from a `symbolt` | the 179 `symbol_expr(symbolt)` sites | Trivial convenience helper (Part II §5 lists the same gap). |

## 7. Phased, commit-sized decomposition

Ordered soundness-first / lowest-risk-first. Each numbered item is one
reviewable commit; **apply and test one at a time** (incremental patch
testing); do not batch. Every commit is gated by §10.

### Phase 3.0 — Baseline & harness (no behaviour change)
1. Capture the golden **verdict + matched-text** set over the full
   `regression/esbmc-solidity` suite (512 tests; §10) on clean `master`,
   on an **asserts-enabled build** (keeps the `migrate.cpp:386-396`
   cross-check live). Record which tests fail pre-change (platform/solver
   baseline). The suite is `.solast`-driven and needs **no `solc`**
   (§10.1) — confirm `solc` absence does not change the set.
2. Add a `sol_typeinfo` round-trip / equivalence unit harness skeleton
   under `unit/solidity-frontend/` (none exists today): asserts that
   classification computed from JSON equals the legacy `#sol_*` value for
   a corpus of nodes. This is the durable contract regression for 3.1.

### Phase 3.1 — De-attribute: move Solidity semantics off the IR (THE key phase)
*No change to emitted IR shape; pure relocation of metadata. This is the
soundness hardening and the prerequisite for every later phase.*

3. Introduce `sol_typeinfo` + the symbol-keyed side-table and the
   threading parameter; populate it wherever a `#sol_*` is currently
   `set`. Leave the `#sol_*` writes in place (dual-write) so reads are
   unaffected.
4. Migrate **`#sol_type` reads** (~43 sites) from `get_sol_type(typet)`
   to the companion. Do this in dependency order: type/decl converters
   first, then expression/call converters. After the last read is
   migrated, delete the `#sol_type` write and the `set_sol_type`/
   `get_sol_type` helpers (`solidity_convert.h:68-75`).
5. Migrate `#sol_contract`, `#sol_data_loc`, `#sol_state_var`,
   `#sol_name`, `#sol_mapping_array`, `#sol_dynarray_state`,
   `#sol_tuple_id` reads → companion; delete each write once its last read
   is gone (one commit per attribute or tight cluster).
6. Fold `#sol_array_size` into the real `array_type2t.array_size` path
   (it is genuine type shape, not metadata) and drop the attribute.
7. Decide and apply the `bytesN` representation (Q3); move
   `#sol_bytesn_size` into `sol_typeinfo.bytesn_width`; drop the attribute.
8. Reclassify the markers `#is_sol_virtual`, `#is_sol_override`,
   `#inlined`, `#sol_unchecked`, `#member_name`, `#cpp_type`/`cpp_type`,
   `#is_modifier_placeholder`, `#zero_initializer`: either move to the
   companion (if read) or delete (if set-only / dead — several are, per
   §4). Each deletion is a branch-shape change → C-Dead discharge or a
   cited zero-read grep.

*Exit 3.1: `grep -rn '"#sol' src/solidity-frontend` returns only genuine
type-shape uses already mapped to IREP2 fields, or nothing. `migrate_type`
applied to any Solidity-produced type now loses no information.*

### Phase 3.2 — Enabling infrastructure (§6 gaps)
9. Add the `symbol2tc`-from-`symbolt` helper and the
   `sideeffect2t(function_call)` / `code_function_call2t` builder wrappers
   the converter will use, with unit tests. No converter call sites change
   yet.
10. Resolve the string/`bytes` literal decoder question (Q4): either port
    `mb_value` onto `constant_string2t` (verbatim; R10) or fix the seam to
    keep literal lowering legacy. Land whichever with property tests
    against the existing CPython-style oracle where applicable.

### Phase 3.3 — Migrate type construction to `type2tc` (internal)
11. Rewrite `get_type_description` / `get_elementary_type_name` /
    `get_parameter_list` and the `solidity_convert_type.cpp` builders to
    produce `type2tc` (companion alongside), back-migrating to `typet`
    only at `get_default_symbol`/`move_symbol_to_context`. One converter
    family per commit (elementary → array/mapping → struct/contract →
    function/code types). `code_type2t` requires `args.size() ==
    argument_names.size()` (`irep2_type.h:229-241`) — supply names.

### Phase 3.4 — Migrate expression construction to `expr2tc` (internal)
12. Rewrite the expression converters (`solidity_convert_expr.cpp`,
    `_ref`, `_call`, `_mapping`, `_tuple`, `_literals`) to build `expr2tc`
    via typed factories instead of `exprt`/`symbol_expr`/
    `side_effect_expr_function_callt`/`member_exprt`/`index_exprt`/
    `address_of_exprt`/`typecast_exprt`. `solidity_convert_call.cpp` is
    the hotspot (98 of 124 `move_to_operands`); split it across several
    commits by handler. Back-migrate expressions to `exprt` only where
    they are stitched into a legacy `codet` body (Phase 3.5 boundary).

### Phase 3.5 — The body boundary (what stays legacy, made explicit)
13. Localize the legacy hand-off: statement assembly
    (`solidity_convert_stmt.cpp`) keeps emitting structured legacy `codet`
    (P1), but its *operands* are now IREP2 lowered via a single
    `migrate_expr_back` helper at the point each statement is built.
    Document this as the durable Solidity frontend boundary — the analogue
    of Part I's `migrate_expr`-at-goto-convert seam.

### Phase 3.6 — Tighten & census
14. Remove now-dead legacy includes and helpers (`typecast.cpp` wraps
    `c_typecastt` on legacy `exprt` — keep only if still on the legacy
    side of the seam; otherwise retire). Snapshot the Solidity IREP2-share
    census and record it here as Part III's outcome section, mirroring
    Part II §11.

### Out of scope — separate tracking issues (the deep pins)
- **IREP2-native goto-convert input** (structured control-flow code
  kinds + an IREP2 `goto_convert_rec`). Removes P1. Large; affects every
  frontend.
- **IREP2-native `clang_cpp_adjust` / `c_link`** (or a Solidity-specific
  adjust that does not round-trip through legacy). Removes P2. Shared
  infrastructure.
- **IREP2-native C printer for Solidity counterexamples** (Part II Phase
  2.7). Witness/`test.desc` text parity bar.

Until those three land, the Solidity frontend's *bodies* and the
*adjust/link* post-processing remain legacy, exactly as for every other
frontend. Part III's deliverable is the de-attributed, IREP2-internal
converter with a single audited legacy seam — genuine forward progress
that is fully behaviour-preserving.

## 7.1 Acceptance criteria per phase

| Phase | Done when |
|---|---|
| 3.0 | Golden verdict+text set captured on asserts build; `solc`-free run reproduces it; `sol_typeinfo` unit harness compiles and passes against legacy `#sol_*` values. |
| 3.1 | `grep '"#sol' src/solidity-frontend` empty (or only IREP2-field-backed shape); a deliberate `migrate_type`→`migrate_type_back` round-trip of any Solidity type in a unit test loses no classification; full suite verdict+text set unchanged; dual-solver agreement. |
| 3.2 | New factories/helpers have round-trip unit tests; no converter call site changed; suite unchanged. |
| 3.3 | Type converters return `type2tc`; back-migration confined to symbol store; suite verdict+text set unchanged on both solvers; asserts-build cross-check silent. |
| 3.4 | Expression converters build `expr2tc`; legacy `symbol_expr`/`side_effect_expr_function_callt`/`*_exprt` count in `src/solidity-frontend` strictly decreasing PR-over-PR; suite unchanged. |
| 3.5 | Exactly one documented `migrate_expr_back` hand-off feeds legacy `codet` bodies; structured CF still lowered by `goto_convert`; suite unchanged. |
| 3.6 | Census recorded; dead legacy retired with C-Dead discharge; suite unchanged. |

The bar for **every** phase: identical pass/fail set and identical
matched output text under **both Bitwuzla and Z3**, on an asserts-enabled
build, over the full `esbmc-solidity` suite.

## 8. Dependencies and prerequisites

- **3.1 blocks everything.** No node may migrate while it still carries
  semantics that `migrate_type`/`migrate_expr` would drop (F4). This
  ordering is the single most important correctness decision in the plan.
- **3.2 blocks 3.3/3.4** (factories must exist and be tested first — the
  V-track lesson from Part I: land the dead-but-tested infrastructure as
  its own PR so the wiring PR has zero coverage-axis risk).
- **3.3 blocks 3.4** (expressions carry their `type2tc`; types must be
  buildable first).
- **3.5 depends on P1 staying pinned.** If the out-of-scope IREP2
  goto-convert ever lands, 3.5's hand-off is where it would connect.
- **3.4's string/`bytes` work depends on Q4** (mb_value).
- The whole plan depends on the **goto-binary on-disk format not
  changing** (Part I, B4) and on the `.solast` input contract (§11).

## 9. Risk register (Solidity-specific; extends Part II §7)

| # | Area | Risk | Sev | Mitigation |
|---|------|------|-----|------------|
| RS1 | **soundness** | Attribute drop (F4): any `#sol_*` surviving into a migrated node silently loses Solidity semantics → wrong operational model selected → **silent wrong verdict**. | **critical** | Phase 3.1 first; unit test that round-trips every Solidity type through `migrate_type`/`_back` and asserts the companion is sufficient; `grep '"#sol'` empty as a CI gate. |
| RS2 | soundness | **Data location** (`#sol_data_loc`): losing `storage` vs `memory` collapses an aliasing write into a local copy (or vice-versa) → missed state-mutation bug (false SUCCESSFUL) or spurious failure. | critical | Model `storage`/`memory`/`calldata` explicitly in `sol_typeinfo`; add focused regression contracts exercising storage-alias vs memory-copy that must keep their current verdicts. |
| RS3 | soundness | **Type-class confusion** (`#sol_type` MAPPING vs DYNARRAY vs BYTES): drives which c2goto operational model is called; a wrong class emits wrong-semantics code. | critical | 3.1 migrates reads in dependency order with the verdict+text gate after each; the 512-test suite already exercises mapping/array/bytes heavily (`mapping_*`, `bytes_*`, `abi_*`). |
| RS4 | soundness | **`bytesN` width loss** (`#sol_bytesn_size`): wrong masking/comparison width on fixed-byte types. | high | Decide one `bytesN` representation (Q3); width lives in `sol_typeinfo`; `bytes_*` regression set is the oracle. |
| RS5 | soundness | **256-bit arithmetic & overflow**: `uint256`/`int256` must map to `*bv_type2t(256)` and overflow/`unchecked` (`#sol_unchecked`) must wrap exactly; a width or check-suppression error flips overflow verdicts. | high | Verify width plumbing end-to-end; keep `--overflow-check` / `--unsigned-overflow-check` Solidity tests green; preserve the `in_unchecked_block` suppression path (`solidity_convert.h:861`). |
| RS6 | soundness | **Overload binding** between name-identical legacy/IREP2 builders (`gen_zero`, `member`, `symbol_expr` vs `symbol2tc`) — both compile, wrong one binds. | med | Per-site review; lean on `-Werror`; unit equivalence tests (Part II R12). |
| RS7 | correctness | **`code_type2t` arity assertion** (`assert(args.size()==names.size())`, `irep2_type.h:240`) — Solidity sometimes builds function types without names; will assert at construction. | med | Supply synthesized parameter names in 3.3; covered by asserts build. |
| RS8 | compatibility | **Counterexample text**: ~21 % of Solidity tests match more than the bare verdict (function names, assertion-coverage counts, line numbers). A construction change that alters symbol naming or pretty-printed types breaks `test.desc` regexes. | high | §10 captures matched text, not just verdict; diff per phase; keep the legacy C printer for Solidity (Part II 2.7 out of scope). |
| RS9 | scope-creep | The "real" migration tempts touching P1/P2 (goto-convert / adjust). Doing so inside this plan blows the blast radius across all frontends. | med | P1/P2 are separate tracking issues; Part III's seam (3.5) stops at the body boundary. |
| RS10 | process | `clang_cpp_adjust` already special-cases Solidity (`solidity_language.cpp:362-380` saves/restores sol64 bodies); a construction change can perturb what adjust mutates. | med | Keep the save/restore logic; re-verify the intrinsic + sol64 library bodies are byte-identical post-adjust. |

## 10. Validation, regression & equivalence strategy

Reuse the Part II universal gate, specialized for Solidity. The suite is
`regression/esbmc-solidity` — **512 test directories**, label
`esbmc-solidity` (registered when `-DENABLE_SOLIDITY_FRONTEND=On`;
`regression/CMakeLists.txt`). Distribution: ~312 CORE, ~192 THOROUGH
(Linux-only), ~8 KNOWNBUG. Per-test timeout default 1200 s.

**Per-phase gate (all must hold, both Bitwuzla and Z3):**
1. **Verdict set identical** to the 3.0 baseline — same `VERIFICATION
   SUCCESSFUL`/`FAILED` per test. Verdicts are immune to the
   model-naming nondeterminism that makes `diff_goto_baseline` unreliable
   across rebuilds (Part II §8.1), so this is the primary oracle.
2. **Matched-text identical.** ~21 % of `test.desc` files assert more than
   the verdict — function names (`github_497_*`: `function func_sat`),
   assertion-coverage counts (`sol_cov_*`: `Total Asserts:`,
   `Assertion Instances Coverage: N%`), line numbers. Capture and diff the
   *matched* lines, not just the verdict. This is the backstop for RS8.
3. **Asserts-enabled build** so the `migrate.cpp:386-396` symbol round-trip
   cross-check stays live across the corpus — it is the strongest
   evidence that whatever the converter now produces is still losslessly
   convertible.
4. **Affected unit suites** green, including the new
   `unit/solidity-frontend/` `sol_typeinfo` equivalence tests.
5. **Run subset first, full last.** CORE subset (`ctest -L esbmc-solidity
   -R '<subset>'`) on every commit; the full 512 (THOROUGH included) on
   phase close. Respect the project 5-minute full-suite cap — narrow scope
   per commit; the full THOROUGH run is a phase-boundary activity.

**Phase-3.1-specific equivalence test (the load-bearing one):** a unit
test that, for a representative corpus of solc JSON type nodes, asserts
`classify_from_json(node) == legacy_#sol_type(node)` for *every* node, and
that `migrate_type_back(migrate_type(built_type))` followed by re-reading
the companion yields the same `sol_typeinfo`. This pins RS1.

### 10.1 `solc` and the `.solast` input contract
The suite ships pre-generated `.solast` JSON (482/512 dirs) and reads it
directly (`test.desc` line 2); the `--sol contract.sol` flag is
informational. **No `solc` binary is required to run the regression
suite** — confirmed: `solc` is absent in this environment and the suite is
designed to run off committed `.solast`. A frontend refactor is therefore
validated entirely on committed inputs; regenerating `.solast` is a
separate upstream concern. Do **not** regenerate `.solast` as part of this
migration (it would conflate solc-version drift with refactor effects).

## 11. Backward-compatibility considerations

- **Goto-binary on-disk format: unchanged** (Part I, B4 / RETAIN_BOUNDARY).
  Old `.goto` binaries must still load; the Solidity path produces the
  same serialized IR.
- **`.solast` input format: unchanged.** The converter still consumes solc
  `--ast-compact-json`; §10.1.
- **CLI surface unchanged:** `--sol`, `--contract`, `--function`,
  `--focus-function`, `--solc-bin`, `$SOLC` (`solidity_language.cpp:58-79,
  116-147`).
- **Counterexample / diagnostic text: must be preserved** where
  `test.desc` matches it (RS8). The legacy C printer stays (Part II 2.7
  out of scope), so pretty-printed types and symbol names are stable.
- **Operational models (sol64, `src/c2goto/library/`):** unchanged; the
  converter still emits calls into them. The model-selection *logic* is
  what 3.1 hardens (it must pick the same model), but the models
  themselves are untouched.

## 12. Success metrics and exit criteria

- **Primary (soundness):** `grep -rn '"#sol' src/solidity-frontend`
  reaches empty (or only IREP2-field-backed shape), proving no Solidity
  semantics ride on droppable irep attributes. This is the metric that
  matters; it closes RS1–RS4 by construction.
- **Secondary:** the legacy-builder census in `src/solidity-frontend`
  (`symbol_expr`, `side_effect_expr_function_callt`, `*_exprt`, `code_*t`
  for non-control-flow) strictly decreasing PR-over-PR; the Solidity
  IREP2-share rising from <1 % (Part I surface table).
- **Exit (this plan):** the converter computes Solidity semantics in the
  typed companion, builds types and statement-operand expressions in
  IREP2, and hands off to `clang_cpp_adjust`/`goto_convert` through one
  documented legacy seam (3.5). Function-body control-flow lowering
  (P1), the shared adjust/link passes (P2), and the Solidity C printer
  remain legacy under their own tracking issues. Verdict+text parity
  holds across the full 512-test suite on both solvers.

The bar is Part I's bar: **behaviour-preserving at every step, proven,
not assumed** — and for a smart-contract verifier, "proven" specifically
means no `#sol_*` attribute can ever be silently dropped on the way to
the solver.

## 13. Open questions (resolve before the cited commit)

- **Q-S1** — Is the symbol-`id`-keyed `sol_typeinfo` side-table sufficient,
  or are there `#sol_*` reads on sub-expression types that never become a
  named symbol (so need JSON-derived classification instead)? Audit the
  ~43 `#sol_type` read sites in 3.1 commit 4 and bucket each as
  "symbol-reachable" vs "must reclassify from JSON." (blocks 3.1)
- **Q-S2** — Does `clang_cpp_adjust` read any `#sol_*` attribute itself
  (i.e. is any Solidity semantics consumed *after* the converter, in the
  shared pass)? If yes, that attribute cannot be made purely
  frontend-local and the companion must outlive the converter. Grep
  `clang-cpp-frontend` for `#sol`. (blocks 3.1 / re-scopes P2)
- **Q-S3** — Canonical `bytesN` representation: `8N`-bit `unsignedbv` vs
  the existing `bytesN` operational-model struct? Whichever the c2goto
  models expect; must match to keep `bytes_*` verdicts. (blocks 3.1
  commit 7)
- **Q-S4** — Can `constant_string2t` carry the `mb_value` decode the
  Solidity string/`bytes` literals need, or must literal lowering stay
  legacy at the seam? (Part II R10/2.6) (blocks 3.4 string work)
- **Q-S5** — Do any Solidity `test.desc` regexes match counterexample
  *values* (not just verdict/function-name/coverage)? Sample-grep before
  3.4; if so, symbol-naming stability becomes a hard 3.4 gate. (sets the
  RS8 bar)
- **Q-S6** — Are `#is_sol_virtual` / `#is_sol_override` / `#inlined` /
  `#sol_tuple_id` ever read, or purely set-only (dead)? §4 shows several
  with 0 read sites; confirm per-attribute before deleting vs relocating.
  (blocks 3.1 commit 8)

## 14. Outcome — Phase 3.1 concluded; the chokepoint milestone and the value-carried-metadata wall

> **Status: concluded.** Sections 1–13 above were the forward plan. This
> records what landed, the resolution of the gating open questions (Q-S1,
> Q-S2), and why the off-IR companion step is deferred. The
> attribute-access **encapsulation chokepoints are the sound milestone**;
> moving the metadata off the IR hits a fundamental "metadata must travel
> with the value" constraint that re-validates B1.

### What landed — the attribute-chokepoint pass

Every **Solidity-specific `#sol_*` type attribute** is now written and
read through a single typed accessor on `solidity_convertert`
(`set_`/`get_`/`has_sol_*`), instead of scattered raw `irept::set`/
`get("#…")` calls.

| PR | Attribute(s) | sites funnelled |
|----|------|---|
| [#4951](https://github.com/esbmc/esbmc/pull/4951) | `#sol_array_size` | 8 set / 13 read |
| [#4952](https://github.com/esbmc/esbmc/pull/4952) | `#sol_bytesn_size` | 5 set / 12 read |
| [#4953](https://github.com/esbmc/esbmc/pull/4953) | `#sol_contract` | 1 set / 6 read |
| [#4958](https://github.com/esbmc/esbmc/pull/4958) | `#sol_mapping_array`, `#sol_dynarray_state`, `#sol_state_var`, `#sol_name` | 7 set / 15 read |
| [#4959](https://github.com/esbmc/esbmc/pull/4959) | `#sol_data_loc` | 3 set / 0 read |

(`#sol_type` was already encapsulated pre-plan via `get_sol_type` /
`set_sol_type`.) After the pass, raw `"#sol` access outside the accessor
bodies is gone; the Solidity type-metadata surface sits behind ~10 typed
seams that a future migration can repoint **in one place each**, rather
than at the dozens of scattered call sites that existed before. Every PR
was a behaviour-preserving mechanical no-op (same key, same value,
same storage); verdict validation ran on Linux CI, since the
`esbmc-solidity` suite cannot run on macOS (the `sol64` operational models
are stubbed empty there — `_BitInt(256)` is unavailable on Apple's aarch64
target).

`#member_name` was **deliberately excluded** — see Q-S2.

### Q-S1 resolved — the reads are value-carried, not symbol-reachable

The plan gated the off-IR companion (a `symbol-id → sol_typeinfo`
side-table) on Q-S1. The audit settles it, and the answer kills the
side-table design.

Reproducible census
(`grep -hoE 'get_sol_type\([^)]*\)' src/solidity-frontend`): of the ~43
`get_sol_type` read sites, **~40 read off a transient `typet` value or an
expression's `.type()`** — local `t` / `val_t` / `base_t` / `rt`,
`comp.type()`, `lhs.type()` / `rhs.type()`, `type.return_type()` — with
**no associated symbol at read time**. The canonical shape
(`solidity_convert_decl.cpp`):

```cpp
typet t;
get_type_description(ast_node, …, t);           // builds t, tags it via set_sol_type
bool is_contract = get_sol_type(t) == CONTRACT; // reads it back off the local t — before any symbol exists
```

There is **no stable key** to look these up by: `typet` is a value with no
identity (copies compare equal but are distinct objects), and the
originating solc JSON node is gone at most read sites. So a side-table
keyed by symbol id — or by anything else — cannot serve them. **The
classification must travel with the type value.**

### Q-S2 resolved — `#member_name` is a shared clang-cpp convention, not Solidity-private

`#member_name` is set by the Solidity frontend on struct-component types,
but it is **read outside the frontend** by
`clang-cpp-frontend/clang_cpp_adjust_code_gen.cpp` (constructor namespace
lookup / name mangling — `clang_cpp_adjust` runs over Solidity output too)
and referenced by the python frontend. It is therefore *not* a
Solidity-local attribute and cannot be moved off the IR by frontend work;
the shared adjust pass reads it straight from the node. Left on the IR,
out of scope. This also answers Q-S2 in the affirmative for at least one
attribute: a shared post-converter pass *does* consume an IR attribute the
Solidity frontend writes.

### The design implication — only a value-bundled wrapper can move it off-IR

| Mechanism | Carries classification with the value? | Verdict |
|---|---|---|
| Side-table (symbol- or otherwise-keyed) | no — transient / expr types have no key | **not viable** (Q-S1) |
| Value-bundled wrapper `struct { type2tc ir; sol_typeinfo info; }` as the frontend's internal type, lowered to plain `type2tc` at the symbol / expr boundary | yes | **works, with a caveat** |
| Extend `type2t` with a generic attribute map | yes | **rejected** (§5.1) — reinstates the string-keyed escape hatch IREP2 exists to abolish |

The only off-IR-capable option is the wrapper. But the wrapper
**re-implements `irept`'s value-traveling attribute flexibility on top of
`type2tc`** — out of band, yet functionally the very thing the migration
set out to remove. It would buy typed-construction safety inside the
converter, but not the determinism / closed-type-system benefit that
justified migrating in the first place; that benefit is already realised
in the verifier core, which never sees these attributes (they are dropped
at the `migrate_type` seam, harmlessly, because nothing downstream reads
them).

### Why Phase 3.1 stops at the chokepoints

This **re-validates B1** (Part I: "frontends stay legacy"). The Solidity
frontend depends on rich, *value-traveling* type metadata — precisely what
IREP2's closed type system is designed not to carry. Bridging it requires
either the wrapper (high cost; recreates attribute flexibility; no
verifier-core soundness gain) or extending `type2t` (rejected). The
cost/benefit of the off-IR / `type2tc` step is therefore poor, and —
consistent with this document's governing rule that **verification
correctness outranks implementation convenience** — pushing it through
would trade the migration's own rationale for churn.

The encapsulation pass is the milestone: a genuine hardening (no scattered
raw irep attribute access; one typed seam per attribute) that leaves the
frontend in the cleanest state it has been on this axis, with a single
repoint-point per attribute should a future wrapper-based effort be taken
up under its own tracking issue. Q-S3–Q-S6 are moot — they gated phases
3.3–3.4, which are not entered.

### Bottom line

Phase 3.1's **attribute-access encapsulation is complete and is the
stopping point.** Moving Solidity type metadata off the IR is blocked not
by effort but by an architectural mismatch: the metadata is read off
transient type *values*, so it must travel with the value — which only a
`type2tc`-wrapping companion can do, and that companion re-introduces the
attribute flexibility IREP2 removed, for no verifier-core benefit. The
frontend → `type2tc` migration therefore remains, as in B1, **out of
scope**; the durable boundary stays the `migrate_*` seam at goto-convert,
and the Solidity frontend stays legacy by design.

---

# Part IV — Python frontend → IREP2 (forward plan)

> **Status: PAUSED at the frontend seam (2026-06-09).** Phases 4.0–4.3
> landed; Phase 4.4/4.5 (expression construction → `expr2tc`) was carried
> as far as the frontend-only scope soundly allows — many converter
> construction sites now build IREP2 internally — and is **deliberately
> stopped here**, for the reason §16.1's go/stop already anticipated and
> §18 below records in full: **every converter-stage construction is an
> `irept`→`expr2tc`→`irept` round-trip.** The IREP2 node is built, then
> `migrate_expr_back`'d at the statement/return/symbol boundary because the
> consumers stay legacy — function bodies are legacy `codet` that
> `goto_convert` reads (W1/P1), and the symbol table receives legacy. The
> byte-identical legacy IR that flows downstream is *why* every step was
> verdict-preserving. Net IR change through the pipeline: **zero**. The
> round-trips are eliminated only by **Part V Phase V.4** (IREP2 bodies +
> IREP2-aware `goto_convert`) and the symbol flip (V.6) — shared,
> repo-wide work, not frontend-scoped. Active work has therefore moved to
> **Phase V.4** (§V.2). See §18 for the pause record and census.
>
> *(Historical status, pre-pause: in flight — Phases 4.0, 4.1 and 4.3 had
> landed; 4.4–4.6 remained.)* Parts I–III are closed records (Part III
> concluded at the attribute-encapsulation milestone and *re-validated B1*
> for Solidity).
> This part is the *execution record + forward plan* for a second frontend
> — Python — that again deliberately reopens boundary B1 ("Frontend → goto
> input", *Deferred indefinitely* in Part I) **for the Python frontend
> only**. Do **not** track it under the Part I umbrella issue, which closed
> declaring B1 out of scope; this work changes that scope for one frontend,
> deliberately and with evidence.
>
> The progress ledger (§0) is the authoritative status; the per-phase
> descriptions in §7 are annotated LANDED / REMAINING accordingly. When the
> two disagree, §0 wins and §7 is stale — fix §7. Every load-bearing count
> and `file:line` in this part is a **snapshot at the revision this edit
> lands on**; re-run the cited command before acting, because the Python
> frontend is the highest-churn tree in the repo (§1, RP1) and anchors drift
> weekly.

This part is written so an engineer who has **never touched the Python
frontend** can execute it. Every load-bearing claim carries a `file:line`
anchor or a re-verification command; run the command before acting, because
the Python frontend is the **highest-churn tree in the repository** (every
file in §3's inventory was modified within the last three weeks of this
writing) and counts *will* drift.

## 0. Progress ledger (authoritative status)

This ledger is the single source of truth for what has landed. Each row
cites the merge PR(s); the body sections are annotated to match. Re-derive
the census row with the §2 commands before trusting it.

| Phase | What | Status | Evidence |
|---|---|---|---|
| 4.0 | Baseline harness + IREP2 type round-trip equivalence test | **LANDED** | `unit/python-frontend/irep2_type_roundtrip_test.cpp` (9 `TEST_CASE`s, lines 52/104/136/156/188/228/253/289/330); CMake target `irep2_type_roundtrip_test`. PR #4992 (commit `2b85d5a6ce`). |
| 4.1 | Attribute-access encapsulation (`#cpp_type`/`#cformat`/`#member_name`) | **LANDED** | Typed accessors `type_utils::{set,get}_cpp_type`, `{set,remove}_member_name` (`type_utils.h:174-196`); `#cformat` routed through `irept::cformat()`. Raw `.set("#`/`.get("#` in `src/python-frontend` reduced to **2 each** (the accessor bodies). PR #4990 (commit `a99fc39171`). |
| 4.2 | Enabling infrastructure (§6 factories, dead-but-tested) | **LANDED (#4997)** | Both expression-side helpers shipped dead-but-tested in PR #4997 (commit `f621558a3a`, 2026-05-31), **after** the prior reconcile that marked this row PARTIAL — see the §0.1 correction note: `symbol_expr2tc(const symbolt&)` and `side_effect_function_call2tc(return_type, function, arguments)` in `src/util/migrate.{h,cpp}` (decls `migrate.h:48-70`, defs `migrate.cpp:433,441`), each proven `==` to `migrate_expr` of the legacy form by round-trip unit tests `unit/util/migrate.test.cpp:225,246`. The `lower_to_seam` type seam (`type_handler.cpp:46-52`) shipped earlier with the 4.3 commits. **Phase 4.4's gating prerequisite is therefore satisfied — 4.4 is unblocked.** |
| 4.3 | **Type** construction → `type2tc` (internal, lowered at seam) | **LANDED for elementary/float/array/pointer/NoneType-Optional/Callable**; struct families deferred | `type_handler.cpp` builders return `type2tc` then `lower_to_seam(...)`: int 492, uint 508, bool 512, uint256 546, float 486, str/char 587-589, NoneType/Optional 458/464, `build_array` 392, Callable 473-478, list pointees 951/964/1057. PRs #5000 (elementary), #5001 (float), #5002 (array), #5018 (Callable), + pointer-family commit `4ebe00eb2c`. Tuple/Optional **struct** builders are F-P5 seam cases deferred to 4.5 (§15.7). `python_int_typet` width logic **unchanged** (F-P7/RP4). |
| 4.4 | **Expression** construction → `expr2tc` (internal) | **IN PROGRESS — file 1 (`converter_unop.cpp`) complete** | Unblocked by #4997. **File 1 done:** the `not`-on-string subtree landed in #5041 (`strlen(base)==0`) and the dict/list `not`-truthiness comparisons in #5045 (`size==0`, removing a `copy_to_operands` site), each built with `symbol_expr2tc` + `side_effect_function_call2tc`/`equality2tc` + `gen_zero(type2tc)` and lowered with one `migrate_expr_back`, gated dual-solver with the asserts cross-check silent (`unary_not_string{,_fail}`, `list_not_truthiness{,_fail}`, `dict40/43`). **General construction blocked (§16.1, F-P11):** files 2 (`converter_expr.cpp`) and 4 (`converter_binop.cpp`) both hit the resolved-type asserts via independent routes — direct `member2t`/`index2t` construction *and* `migrate_expr` of operands that contain unresolved member/index sub-expressions (an integer-arith trial aborted ~277/432 tests). The migrate-forward/back-migrate recipe is therefore feasible **only for fully synthetic, concrete-typed operands** (file-1 subtrees); general user-expression construction can't migrate until `adjust` resolves types (P2, out of scope). **Recommendation: declare 4.4 complete-as-feasible at the file-1 (synthetic-operand) level** — the same "encapsulate, stay legacy at the seam" end-state Part III reached for Solidity (§16.1 go/stop table). Per-file guide §7.2; subtree-not-site rule §7.2.3. |
| 4.5 | The legacy body seam (one documented `migrate_expr_back` hand-off) | **REMAINING** | Not started. Absorbs the tuple/optional struct builders (§15.7) and `#cpp_type`/`#member_name` re-attachment (F-P5). |
| 4.6 | Tighten, census, go/stop decision | **REMAINING** | Not started. |

**Census at this revision** (re-run §2; figures re-counted 2026-06-02):
legacy IR mentions in `src/python-frontend` **4682**, IREP2 (`*2tc`) **31**
(was 23 at the prior reconcile, 6 pre-4.3 — the continued rise is Phase 4.3's
`type_handler` migration plus the Callable commit #5018). `set_type(2tc)` /
`set_value(2tc)` at the symbol-write boundary: still **0** — i.e. types are
built in IREP2 *internally* and back-migrated at `lower_to_seam`; the symbol
table still receives legacy `typet` (F-P1). `migrate_{expr,type}` mention
count rose from 3 to **10**, all in `type_handler.cpp` (the `lower_to_seam`
seam) and `function_call/str_conv.cpp:559,573` (pre-existing simplifier
round-trip). This is the expected mid-migration shape: the IREP2 surface
grows *inside* the type builders while the seam back-migrates so externally
observable bytes stay identical.

## 0.1 Reconciliation correction (2026-06-02) — Phase 4.2 had already landed

> Recorded here in the §15.1 spirit: a stale status in a verification-planning
> document is the same class of defect the plan exists to prevent, so the
> correction is logged rather than silently overwritten.

A re-audit on 2026-06-02 (`git log -L 60,70:src/util/migrate.h`) found that
**Phase 4.2 landed in PR #4997** (commit `f621558a3a`, 2026-05-31) — *before*
the most recent doc reconcile (#5037, 2026-06-02) — yet that reconcile still
marked 4.2 **PARTIAL** with its factories "not yet landed" and gating Phase
4.4. The reconcile missed #4997. The two expression-side helpers the plan
specified are present, named, and unit-tested:

| Planned (anticipated) name | Landed name + location | Test |
|---|---|---|
| `symbol2tc`-from-`symbolt` | `symbol_expr2tc(const symbolt&)` — `migrate.h:61`, `migrate.cpp:433` (`return symbol2tc(migrate_symbol_type(sym), sym.id);`) | `unit/util/migrate.test.cpp:225` — asserts `symbol_expr2tc(sym) == migrate_expr(symbol_expr(sym))` and round-trips |
| `sideeffect2t(function_call)` wrapper | `side_effect_function_call2tc(return_type, function, arguments)` — `migrate.h:67`, `migrate.cpp:441` (`sideeffect2tc(..., allockind::function_call)`) | `unit/util/migrate.test.cpp:246` — asserts `==` to `migrate_expr` of a real `side_effect_expr_function_callt` |

**Consequences propagated through this part:** §0 ledger (4.2 → LANDED, 4.4 →
unblocked), §4.2 prose, §6 gap table (two rows → LANDED), §7.2/§7.2.1
(prerequisite satisfied; idiom table now cites the *actual* helper names, not
the anticipated ones), and §8 dependencies. The helpers are **dead-but-tested**
— `grep -rn 'symbol_expr2tc\|side_effect_function_call2tc' src/python-frontend`
→ empty — exactly the V-track shape the plan asked for, so Phase 4.4 begins by
*wiring* existing tested infrastructure, not by building it. (Update: the
first wiring commit has since consumed both helpers in `converter_unop.cpp`, so
that grep is no longer empty — see the §0 ledger 4.4 row.)

## 1. Scope, motivation, and the non-negotiable constraint

`src/python-frontend/` is ~55 kLOC and growing (the converter alone —
`converter/*.cpp` — is ~440 kB; `python_list.cpp` is 4905 lines,
`python_dict_handler.cpp` 3302). It was **100 % legacy IR** at the start of
this work; after Phase 4.3 (§0) a census
(`git grep -P '\b([A-Za-z_]*(exprt|typet|codet)|irept)\b' -- src/python-frontend`
vs the `*2tc` family) returns **≈4682 legacy mentions to ≈23 IREP2** — still
the lowest IREP2 share of any frontend after Solidity, but no longer zero.
The **expression** converters build `exprt`/`codet` and write them into
`symbolt` through the **legacy** setters; the symbol-write boundary never
calls `set_type(type2tc)`/`set_value(expr2tc)` (verified: `grep -rn
'set_type(\|set_value(' src/python-frontend | grep -c 2tc` → **0**). The
**type** builders (`type_handler.cpp`, post-4.3) now construct `type2tc`
internally and back-migrate at one seam (`lower_to_seam`, §0) — so the
table still receives a legacy `typet` and external bytes are unchanged,
which is exactly the "IREP2-internal, legacy-at-the-seam" target of §5.

Why Python is a defensible second pilot *after* the Solidity result:

- Unlike Solidity, the Python frontend carries **almost no Solidity-style
  value-traveling irep metadata**. The entire `#`-attribute surface is
  three keys — `#cpp_type`, `#cformat`, `#member_name` — versus Solidity's
  ≥20 `#sol_*` (Part III §4 F4). Python type classification lives in the
  **JSON AST annotations** (`python_annotation`), is consumed at
  construction time, and is *not* re-read off irep nodes. So the central
  Solidity soundness hazard (RS1: silent attribute drop at `migrate_type`)
  is structurally far smaller here — see F-P4.
- The frontend does **not** invoke `c_link` (confirmed:
  `grep -rn c_link src/python-frontend` → empty). One of the two shared
  legacy post-passes that pinned Solidity (Part III F3/P2) is simply
  absent. Only `clang_cpp_adjust` runs over Python output
  (`python_language.cpp:245`).
- It uses typed factories *and* substantial raw irep operand surgery —
  see the **corrected** census in §15 (an earlier draft of this bullet
  reported `0` operand-surgery sites; that was a `grep -E '\|'`
  false-negative). The real figures are **576** `move_to_operands` /
  `copy_to_operands`, **166** `.operands()`, **142** `.id()`, **161**
  `.find(`, concentrated in the operational-model handlers
  (`python_list.cpp` 101, `python_set.cpp` 83, `python_dict_handler.cpp`
  75, `converter_stmt.cpp` 74). This is **more** entangled than Solidity
  (which had ~124 `move_to_operands` total), so Phase 4.4 is correspondingly
  harder, not easier — the per-handler decomposition in §7 and risk RP13
  (§9) reflect this. Re-verify with the proper-ERE commands in §15.

**Counterweight — why this is harder than Solidity in other dimensions
(read before committing to the project):**

- **Scale and churn.** Solidity is ~23.5 kLOC and quiescent; Python is
  ~55 kLOC and changes daily. A long-running migration branch will face
  constant merge conflict against a moving target. This is the single
  biggest *project-management* risk and it is not technical (§9 RP1).
- **The same hard pins survive.** P1 (function bodies are legacy `codet`;
  IREP2 has no structured control-flow code kinds — Part III F2) and the
  shared-pass pin P2 both still apply. Python emits `code_ifthenelset`,
  `code_breakt`, `code_continuet`, `code_labelt` (re-verify:
  `grep -roE 'code_[a-z]+t' src/python-frontend/converter | sort -u`) —
  all structured forms `goto_convert` must lower. The frontend therefore
  **cannot** emit IREP2 bodies; bodies stay legacy `codet` exactly as for
  every other frontend.
- **`#cpp_type` and `#member_name` are read by the *shared* pass.**
  `clang_cpp_adjust_expr.cpp:464` does `type.get("#cpp_type")`, and
  `#member_name` is the shared clang-cpp struct-component convention
  (Part III Q-S2). Neither can be moved off the IR by frontend-only work —
  the post-converter adjust pass reads them straight off the legacy node.
  This is the Python analogue of the wall Part III hit (§14, and F-P5).

**The constraint is Part I's constraint, restated and non-negotiable:**
ESBMC is a verifier; **every step must be behaviour-preserving** —
identical pass/fail verdicts, identical counterexample-visible text where
`test.desc` matches it, dual-solver agreement, no on-disk goto-binary
format change. A step that is merely "more IREP2" but shifts one Python
verdict is a regression, not progress. Where this plan and implementation
convenience conflict, **correctness wins** — the user has made this the
governing rule.

## 2. How this plan was derived (reproducibility)

Same method as Part II §2 / Part III §2, re-run against the Python tree.
Each command is re-runnable; counts are a snapshot at the commit this
document lands on.

- **IR census** —
  `git grep -P '\b([A-Za-z_]*(exprt|typet|codet)|irept)\b' -- src/python-frontend | wc -l`
  (legacy) vs `git grep -P '\b[A-Za-z_]+2tc?\b' -- src/python-frontend | wc -l`
  (IREP2).
- **Builder-idiom census** — per-idiom counts via
  `grep -rE '<idiom>' src/python-frontend | wc -l` for `symbol_expr(`,
  `side_effect_expr_function_callt`, `member_exprt`, `index_exprt`,
  `address_of_exprt`, `typecast_exprt`, `constant_exprt`, `code_*t`,
  `set_type(`, `set_value(`. Snapshot below (F-P3).
- **Attribute census** —
  `grep -rhoE '\.(set|get)\("#[a-zA-Z_0-9]+"' src/python-frontend | sort | uniq -c`.
- **Seam trace** — read `symbol.{h,cpp}`, `migrate.{h,cpp}`,
  `python_language.cpp`, `clang_cpp_adjust*.cpp`,
  `goto_convert_functions.cpp` to fix where legacy becomes IREP2.
- **Shared-pass read trace** —
  `grep -rn '#cpp_type\|#cformat\|#member_name' src/clang-cpp-frontend src/goto-programs`
  to find which Python-written attributes are consumed *outside* the
  frontend.
- **IREP2 API audit** — read `irep2_expr.h`, `irep2_type.h`,
  `expr_kinds.inc`, `type_kinds.inc`, `c_types.h` for available factories
  and gaps.
- **Adversarial re-check** — every load-bearing fact in §4 must be
  re-derived by an independent pass instructed to refute it before it is
  acted on; the figures below are the post-correction ones at writing time.

## 3. Current Python frontend architecture and the legacy→IREP2 seam

The Python pipeline is **JSON-mediated**, which is the defining structural
difference from the clang frontends (no in-memory compiler AST objects flow
through it):

```
file.py
  │  python_languaget::parse  (python_language.cpp:80)
  ▼  spawns `python3 parser/__main__.py`  (ast2json) — EXTERNAL process
JSON AST (nlohmann::json `ast`)
  │  python_annotation<json>::add_type_annotation  (python_language.cpp:206)
  ▼  decorates the JSON with inferred types — classification lives HERE,
     in JSON, never on an irep node
annotated JSON AST
  │  python_languaget::typecheck  (python_language.cpp:228)
  │    add_cprover_library(context)         ← C operational models loaded
  │    python_converter::convert()          ← JSON → legacy symbolt table
  ▼
symbolt table (LEGACY-valid: set_type(typet)/set_value(exprt))
  │  clang_cpp_adjust adjuster(context); adjuster.adjust()  (python_language.cpp:245)
  ▼    reads get_value()/get_type() as LEGACY, mutates in place, writes
       LEGACY back; reads `#cpp_type` (clang_cpp_adjust_expr.cpp:464)
  │  (NO c_link — unlike Solidity / C)
  ▼
goto_convert: to_code(symbol.get_value())  ← LEGACY codet body
  goto_convert_rec lowers structured CF (if/while/break/continue/label)
  migrate_expr / migrate_symbol_type per instruction  ──► IREP2 (permanent)
```

The pivot is the symbol table (Part I, B2): `symbolt` stores
`type2tc`/`expr2tc` as source of truth with **lazy legacy caches**
(`symbol.h:89-99`). The Python converter writes the **legacy** side
exclusively — `create_symbol` does `symbol.set_type(type)` on a `typet`
(`converter/converter_util.cpp:21-37`) — so a Python symbol enters the
table legacy-valid / IREP2-invalid, and stays legacy until `goto_convert`'s
`migrate_*` seam converts it for good. This is identical in shape to the
Solidity seam (Part III §3) **minus the `c_link` stage**.

### 3.1 Operational-model surface (the Python-specific complication)

Python's library semantics are supplied by **two** model layers, both of
which the converter targets with `code_function_callt` (135 call sites):

1. **C operational models** — `src/c2goto/library/python/` (`list.c`,
   `dict`-via-`list.c`, `string.c`, `slice.c`, `math.c`, `linalg.c`,
   `umath.c`, `python_types.h`). FLAIL-mangled into `clib*.goto`, loaded by
   `add_cprover_library`. These are **C** → they cross the `migrate_*` seam
   exactly like any C library and are **unaffected** by frontend IR choice.
2. **Python-level models** — `src/python-frontend/models/*.py` (`builtins`,
   `numpy`, `dataclasses`, `decimal`, `collections`, `heapq`, `threading`,
   `re`, `os`, …). These are **Python source** re-parsed through the same
   frontend; they exercise *every* converter path the user program does.

Consequence for validation (§10): the model `.py` files are part of the
input corpus, not external dependencies — any converter change is exercised
against them automatically, but a regression in them is a regression in
*all* programs that import the model.

### 3.2 The front half (parse + type-annotation) is migration-inert — and why this matters

The user's brief asks for "architectural changes required to support irep2
throughout the Python **parsing, type-checking, AST conversion**, and
intermediate-representation generation pipeline." The honest, evidence-backed
answer for the first two stages is **none** — and the reason is structural,
not an omission. Documenting *why* is part of the soundness contract: a
future engineer must not "migrate the parser to irep2" thinking it was
overlooked.

**The parse stage emits zero IR.** `python_languaget::parse`
(`python_language.cpp:80`) spawns an external `python3` running
`parser/__main__.py` (ast2json), reads the emitted JSON
(`python_language.cpp:187`, `ast = nlohmann::json::parse(...)`), and stores
it as an `nlohmann::json` member. No `exprt`/`typet`/`irept` exists at this
point — the representation is JSON, and the JSON contract is frozen (§11,
backward-compatibility: do **not** touch the parser as part of this
migration; that would conflate ast2json drift with refactor effects, the
analogue of Part III's "do not regenerate `.solast`").

**The type-checking stage decorates JSON, not IR.** Python type
classification is produced by `python_annotation<Json>`
(`python_annotation.h`), driven from `python_languaget::typecheck` via
`add_type_annotation()` (`python_language.cpp:210-212`). The annotator
**mutates the JSON AST in place**, inserting string-valued `annotation`
nodes (e.g. `{"_type":"Name","id":"int"}`) — it never constructs an `irept`.
Verified: `grep -rn 'exprt\|typet\|irept\|2tc' src/python-frontend/python_annotation*`
finds the type *names* only as JSON string handling, no IR construction.
This is the structural difference from Solidity, whose classification rode
`#sol_*` attributes **on irep nodes** (Part III §4 F4). For Python the class
lives in JSON and is consumed exactly once, at AST→symbol conversion, by
`type_handler::get_typet` (the §3 seam). **Re-check (the load-bearing
assertion of this whole part):**
`grep -rn 'get("#py\|set("#py\|#python_type' src/python-frontend` → empty
(F-P4) — there is no Python type tag on any irep node to drop at
`migrate_type`.

**Consequence — the migration boundary is the AST→IR converter, not the
pipeline front.** The only stages that build IR are (a) `type_handler`
(types — Phase 4.3, landed) and (b) the expression/statement converters
(Phase 4.4–4.5, remaining). Parsing and type-annotation are **outside the
migration by construction** and must be left byte-for-byte unchanged; the
JSON they produce is the migration's *input contract*, pinned by §11. If a
later engineer believes the annotator should emit IREP2, that is a different
project (it would mean moving type inference below the converter) and is
explicitly **not** this plan — file it separately.

> **Investigation note / gap.** The one place this boundary is fuzzy is
> `converter_types.cpp:get_type_from_annotation` (≈line 176-525), which reads
> the JSON `annotation` and calls `type_handler_.get_typet(...)`. It is the
> bridge from JSON-class to `typet`; it is part of the **converter** (IR
> side), so it *is* in scope for Phase 4.3/4.4. Before extending Phase 4.3
> to the remaining type families, confirm by reading that function that no
> JSON-class semantics are silently lost when it picks a `typet` — any
> annotation string the bridge does not handle today is a pre-existing gap
> (SM-class), to be preserved, not fixed (§14).

## 4. Architectural findings that bound this plan

Each is verified; the re-check command is given. Findings are numbered
`F-Pn` to distinguish from Solidity's `Fn`.

**F-P1 — Symbol table is IREP2-source-of-truth with lazy legacy caches; the
converter writes the legacy setters only.** `symbol.h:89-99`,
`converter/converter_util.cpp:21-36` (`create_symbol` does
`symbol.set_type(type)` on a legacy `typet`, line 32). A frontend *may*
legally call `set_type(type2tc)`/`set_value(expr2tc)`; the Python converter
never does *at the symbol-write boundary* — even post-4.3, the IREP2
`type2tc` a builder produces is back-migrated by `lower_to_seam`
(`type_handler.cpp:46-52`) to a legacy `typet` before it reaches
`create_symbol`. Re-check: `grep -rn 'set_type(\|set_value(' src/python-frontend
| grep -c 2tc` → **0** (still). The invariant the migration preserves is
"the *bytes* entering `symbolt` are byte-identical legacy `typet`"; IREP2
lives strictly inside the builders (§5).

**F-P2 — Function bodies are hard-pinned to legacy `codet`; IREP2 has no
structured control flow (PIN P1).** Identical to Part III F2. The converter
emits `code_ifthenelset`, `code_breakt`, `code_continuet`, `code_labelt`,
plus `code_block`/`code_decl`/`code_assign`/`code_function_call`/
`code_return`. Lowering structured CF to goto is `goto_convert`'s job, below
the frontend boundary. Re-check:
`grep -roE 'code_[a-z]+t' src/python-frontend/converter | sort -u` and
`grep -E 'IREP2_EXPR\(code_' src/irep2/expr_kinds.inc`.
→ **Consequence (PIN P1):** the frontend cannot hand IREP2 bodies to
`goto_convert`. Bodies stay legacy `codet`. Out of scope here.

**F-P3 — Construction is via typed factories, not raw irep surgery (this is
*good* — it makes substitution mechanical).** Snapshot builder census
(re-run the per-idiom `grep -c`):

| Idiom | sites | IREP2 target |
|---|---:|---|
| `symbol_expr(` (converter helper wrapping `symbol_exprt`) | ~844 | `symbol2tc` + a `symbol2tc`-from-`symbolt` helper |
| `typecast_exprt` | 127 | `typecast2t` |
| `member_exprt` | 120 | `member2t` |
| `code_declt` | 144 | stays legacy `codet` body (P1); operand → `expr2tc` |
| `code_function_callt` | 135 | `code_function_call2t` (statement) / stays in body |
| `side_effect_expr_function_callt` | 107 | `sideeffect2t(...function_call)` (`irep2_expr.h:1749`) |
| `code_assignt` | 107 | stays legacy body; operands → `expr2tc` |
| `address_of_exprt` | 85 | `address_of2t` |
| `code_blockt` | 75 | stays legacy body (P1) |
| `constant_exprt` | 42 | typed `constant_int2tc`/`constant_floatbv2tc`/… |
| `dereference_exprt` | 37 | `dereference2t` |
| `code_ifthenelset` | 23 | stays legacy body (P1) |
| `index_exprt` | 21 | `index2t` |
| **`move_to_operands` / `copy_to_operands`** | **576** | typed factory args — but must untangle each call shape |
| **`.operands()` direct access** | **166** | typed fields per `*2t` kind |
| `.id()` string-id branch | 142 | compile-time `is_*2t()` predicates |
| `.find(` raw sub-irep read | 161 | typed fields / no analogue (audit per site) |
| `set_statement("…")` generic codet | 24 | stays legacy body (P1); incl. `while`/`expression` |

The "stays legacy body" rows are the P1 boundary; the rest are
expression-operand construction that *can* move to `expr2tc` internally and
be lowered with one `migrate_expr_back` at the body seam (the Part III §5
target shape). **Caveat (corrected, §15):** the 576 `move/copy_to_operands`
+ 166 `.operands()` + 142 `.id()` + 161 `.find(` sites mean the converters
do *real* string-tree surgery (heaviest in the operational-model handlers),
not just clean factory calls. Each such site must be rewritten to typed
`*2t` field access, which is the bulk of Phase 4.4 and its dominant risk
(RP13). The migration is **not** the mechanical substitution an earlier
draft implied.

**F-P4 — The attribute surface is tiny, and the central Solidity hazard
(silent classification drop) does NOT exist for Python.** Full census:

| Attribute | Sites | Carries | Read where? | Migrates? |
|---|---|---|---|---|
| `#cpp_type` | 5 set / 1 read in-frontend | C-backend float/char width hint | **shared `clang_cpp_adjust_expr.cpp:464`** | dropped by `migrate_type`, but **read pre-migration** |
| `#cformat` | 3 set | numpy literal-format hint | local (numpy fold) | dropped; set-only-ish |
| `#member_name` | 1 set / removed in 1 | struct-component tag | **shared clang-cpp** (Part III Q-S2) | dropped; read pre-migration |

Crucially: **Python does not tag types with a `#py_type` classification the
way Solidity tags `#sol_type`.** The type class is resolved from the JSON
annotation into a *plain* C-like `typet` at construction
(`type_handler::get_typet`), so when a Python type crosses the
`migrate_type` seam there is **nothing semantic to drop** — it is already a
`signedbv`/`floatbv`/`struct`/`array`/`pointer`. Re-check:
`grep -rn 'get("#py\|set("#py\|#python_type' src/python-frontend` → empty.
→ **Consequence:** the Solidity "de-attribute first" phase (Part III 3.1)
is **mostly unnecessary** for Python. The residual `#cpp_type`/`#member_name`
are shared-pass-read and therefore *cannot* be moved by frontend work
anyway (F-P5) — they are pinned, not migrated.

**F-P5 — All three attributes are consumed by shared/downstream passes
outside the frontend (PIN P2), as confirmed by the Q-P3 audit (§15).**
`#cpp_type` is read by **three** consumers: `clang_cpp_adjust_expr.cpp:464`
(shared adjust), `cpp_expr2string.cpp:138-140` (the C++ pretty-printer that
produces counterexample text), and `goto2c/expr2c.cpp:169-174` (the goto→C
emitter). `#member_name` is read by the shared clang-cpp pass
(`clang_cpp_adjust_code_gen.cpp`, `clang_cpp_convert.cpp`). `#cformat` is a
**shared irep infrastructure** attribute (`irept::a_cformat`,
`irep.cpp:507`; `constant_exprt`'s constructor sets it, `std_expr.h:1095`)
read by the pretty-printers for constant formatting. Re-check:
`grep -rn '#cpp_type\|#cformat\|#member_name' src/clang-cpp-frontend src/util src/goto2c`.
→ **Consequence (PIN P2):** even a fully IREP2-internal converter must emit
legacy `typet`/`exprt` carrying these attributes *at the symbol/adjust
boundary*, because adjust, the pretty-printer, and goto2c read them off the
legacy node before/after `migrate_*` (which drops them). All three define
part of the legacy seam and stay there by design, exactly as Solidity's
`#member_name` did (Part III §14).

**F-P9 — The frontend already touches the `migrate_*` seam internally, now
in two distinct loci (count rose 3→10 with Phase 4.3).**
`function_call/str_conv.cpp:559,573` round-trip an operand through
`migrate_expr`/`migrate_expr_back` (the Part II Phase-2.2 IREP2-simplifier
redirect — pre-existing, base-conversion path). Phase 4.3 added the second
locus: `type_handler.cpp`'s `lower_to_seam` (line 48 `migrate_type_back`)
plus the array/pointer builders' `migrate_type` calls (389-390, 951, 964,
1040). So the converter has a **working, tested** IREP2 touch-point in each
of the type and expression worlds — analogous to Part II's note that
`clang_c_adjust_expr.cpp` itself calls `migrate_*`. These are the natural
points Phase 4.4 builds out from. Re-check:
`grep -rE 'migrate_expr|migrate_type' src/python-frontend` → **10**.

**F-P6 — No `c_link`; one fewer shared legacy pass than Solidity/C.**
`grep -rn c_link src/python-frontend` → empty. The only post-converter
shared pass is `clang_cpp_adjust`. This *narrows* P2 relative to Solidity
but does not remove it.

**F-P7 — Python integer semantics are an existing approximation, orthogonal
to IR representation.** `type_handler.cpp:43-51`: default lowering is
`long_long_int_type()` (64-bit, two's complement); `--ir`/bignum mode uses
`signedbv_typet(512)` (`kPythonBignumWidth`). True Python arbitrary
precision is **not** modelled (FIXME `type_handler.cpp:471`). This is a
pre-existing soundness gap (§14 SM1) that the migration must **preserve
exactly, not fix** — changing the width during an IR migration would
conflate two effects and is forbidden by the behaviour-preserving rule.

**F-P8 — The IREP2 construction API is largely present; the gaps overlap
Part II/III's and are enumerable (§6).**

## 5. Target end-state: "IREP2-internal, legacy-at-the-seam" (same as Part III §5)

Given P1 + P2, the **achievable, sound** frontend-scoped target is *not*
"the converter emits IREP2 end-to-end." It is:

> The converter constructs **types and the expression operands of
> statements** using IREP2 typed factories internally, lowering to legacy
> `typet`/`exprt`/`codet` at **one documented seam** — the
> `symbolt`/`clang_cpp_adjust`/body boundary — where the legacy form must
> still carry `#cpp_type`/`#member_name` (F-P5). Structured control-flow
> statements and the function-body `codet` handed to `goto_convert` remain
> legacy by design (P1). The externally observable bytes reaching
> `clang_cpp_adjust` are **bit-identical** to today.

```
              ┌───────────────── frontend scope ─────────────────┐
JSON (typed) ─► type_handler builds type2tc  ─┐
                converters build expr2tc      ─┤ lower at ONE seam
                                               │  (migrate_*_back +
                                               │   re-attach #cpp_type/
                                               │   #member_name)
              └───────────────────────────────┴──────────────────┘
                                               ▼
        symbolt (legacy) ─► clang_cpp_adjust (P2) ─► goto_convert ─► IREP2
```

This buys the IREP2 compile-time-typing soundness benefit *inside the
converter* (a mis-built node fails to compile, not at symex) and leaves
behaviour bit-identical. It is the same deliverable Part III defined for
Solidity — and, per the Part III §14 outcome, the engineer **must
internalize up front** that the deeper boundary (P1/P2) does not move, and
that the honest end state may again be "encapsulate and IREP2-internalize
the construction, with the frontend staying legacy at the seam by design."

> **Rejected alternative — extend `type2t`/`expr2t` with a generic
> attribute map** to carry `#cpp_type`/`#cformat`. This re-introduces the
> string-keyed escape hatch IREP2 was built to abolish (Part I §"Why this
> mattered"; Part III §5.1). Do not. `#cpp_type` stays on the legacy seam
> node instead.

## 6. IREP2 API gaps to close before migrating construction

From the F-P8 audit and reusing Part II §5 / Part III §6. Build each (with
round-trip unit tests) **before** the phase that needs it; each is small.

| Gap | Needed by | Disposition |
|---|---|---|
| `symbol2tc`-from-`symbolt` convenience | the ~842 `symbol_expr(symbolt)` sites | **LANDED (#4997)** as `symbol_expr2tc(const symbolt&)` (`migrate.h:61`, `migrate.cpp:433`): `symbol2tc(migrate_symbol_type(sym), sym.id)`. Unit-tested `==` to `migrate_expr(symbol_expr(sym))` at `unit/util/migrate.test.cpp:225`. Built once, shared across frontends (Part II §5, Part III §6 had the same gap). |
| Expression-context function call | the ~111 `side_effect_expr_function_callt` sites | **LANDED (#4997)** as `side_effect_function_call2tc(return_type, function, arguments)` (`migrate.h:67`, `migrate.cpp:441`): `sideeffect2tc(..., allockind::function_call)` with nil size and empty (not nil) alloctype — the round-trip-stable form (see the in-code note, `migrate.cpp:446-450`). Unit-tested at `unit/util/migrate.test.cpp:246`. |
| `from_double(double, type2tc)` → `constant_floatbv2tc` | float-literal construction (`convert_float_literal.cpp`) | Part II §5 gap; trivial port. |
| `#cpp_type` re-attachment helper at the seam | float/char-typed symbols (F-P5) | **LANDED as `lower_to_seam(type2tc, cpp_type)` (`type_handler.cpp:46-52`):** builds `type2tc`, `migrate_type_back` to `typet`, re-attaches `#cpp_type` via `type_utils::set_cpp_type`. Reused by every 4.3 builder. |
| `mb_value` on `constant_string2t` | `str` / `bytes` literals | **Q-P4 RESOLVED (§13): not needed.** String-literal lowering stays legacy at the seam; `mb_value()` keeps being read off the legacy `string_constantt`, so the R10 verbatim port is not required for the Python migration. |
| 256/512-bit integer types | `--ir` bignum mode (`signedbv_typet(512)`) | `signedbv_type2tc(512)` already works; verify BigInt/SMT width under overflow checks. No new factory. |
| Structured control-flow code kinds | function bodies | **Out of scope (P1).** Bodies stay legacy `codet`. |

## 7. Phased, commit-sized decomposition

Ordered soundness-first / lowest-risk-first. Each numbered item is **one
reviewable commit**; **apply and test one at a time** (incremental patch
testing); do not batch. Every commit is gated by §10. Because the tree
moves fast (§1, RP1), keep each phase short-lived and rebase frequently.

### Phase 4.0 — Baseline & harness (no behaviour change) — **LANDED (#4992)**
1. Capture the golden **verdict + matched-text** set over a stratified
   `regression/python` subset (the suite has **3558 test dirs** — full runs
   exceed the 5-minute cap; §10) on clean `master`, on an **asserts-enabled
   build** (keeps the `migrate.cpp` symbol cross-check live). Record
   pre-existing failures (platform/solver/`ast2json` baseline — note the
   `ast2json` dependency, AGENTS.md "Testing").
2. Add an IREP2-construction equivalence harness under
   `unit/python-frontend/` (extend the existing CMake target there): assert
   that, for a corpus of representative nodes, the `type2tc` a migrated
   builder produces back-migrates (`migrate_type_back`) to a `typet`
   **equal** to the one the legacy builder produces today (modulo the
   `#cpp_type`/`#member_name` re-attachment, which the harness checks
   explicitly). This is the durable contract regression for Phases 4.2–4.3.

### Phase 4.1 — Attribute-access encapsulation (mirror Part III's milestone) — **LANDED (#4990)**
*This is the cheap, high-value, low-risk hardening — done first; it is the
milestone even if no further phase is taken up. The accessors live in
`type_utils.h:174-196`; `#cformat` routes through the standard
`irept::cformat()` mechanism.*
3. Funnel the three `#`-attribute keys through typed accessors on
   `python_converter` / `type_handler` (`set_/get_cpp_type`,
   `set_/get_cformat`, `set_/get_member_name`), replacing the scattered raw
   `.set("#…")`/`.get("#…")` calls (5 + 3 + 1 sites). Behaviour-preserving
   mechanical no-op (same key, same value, same storage). Document that
   `#cpp_type` and `#member_name` are **shared-pass-read (F-P5) and stay on
   the legacy seam node** — they are *not* migration targets. This gives a
   single repoint-point per attribute and matches the Solidity §14 outcome.

### Phase 4.2 — Enabling infrastructure (§6 gaps, dead-but-tested) — **LANDED (#4997)**
*The `lower_to_seam` type seam shipped inside the 4.3 commits (it was the
enabling infra for type lowering). The **expression**-side factories landed
separately in PR #4997 (commit `f621558a3a`) as dead-but-tested infrastructure
— the V-track shape the plan called for — so Phase 4.4 is now unblocked (§0.1).*
4. **DONE (#4997).** `symbol_expr2tc(const symbolt&)` (`migrate.h:61`,
   `migrate.cpp:433`) and `side_effect_function_call2tc(return_type, function,
   arguments)` (`migrate.h:67`, `migrate.cpp:441`) landed with round-trip unit
   tests (`unit/util/migrate.test.cpp:225,246`) and **no converter call-site
   changes** (the Part I V-track lesson: ship tested infrastructure separately
   so the wiring PR has zero coverage-axis risk). Both are proven `==` to
   `migrate_expr` of the legacy constructor they replace. Phase 4.4 wires them.
5. Resolve the string/`bytes` decoder question (Q-P4): port `mb_value` onto
   `constant_string2t` verbatim (R10) *or* decide string-literal lowering
   stays legacy at the seam. Land with property tests against the existing
   CPython oracle (`unit/python-frontend/`, AGENTS.md "Hypothesis tests").

### Phase 4.3 — Migrate **type** construction to `type2tc` (internal) — **LANDED for elementary/float/array/pointer/Callable (#5000–#5018); struct families deferred to 4.5**
6. Rewrite `type_handler`'s builders (`get_typet`, `python_int_typet`,
   `build_array`, the elementary/struct/array/pointer paths) to produce
   `type2tc`, threading any `#cpp_type` need into the seam helper, and
   back-migrating to `typet` only at `create_symbol`/`add_symbol`. **One
   builder family per commit** (elementary → array → struct/class →
   pointer/function). Preserve `python_int_typet`'s width logic byte-for-
   byte (F-P7 / SM1). `code_type2t` requires `args.size() ==
   argument_names.size()` (`irep2_type.h:240`) — supply synthesized names.

   **Ordering refined by the §15.7 byte-identity audit.** The seam helper
   `lower_to_seam` preserves byte-identity *only* when the legacy form
   already carries every field `migrate_type_back` re-emits. The
   **pointer family** (`pointer_typet(value/symbol)`, and the
   NoneType/Optional pointer-width unsignedbv) satisfies this and was
   migrated **ahead of struct/class** (elementary → array → **pointer** →
   …). The **tuple/optional struct builders do *not*** — `migrate_type_back`
   unconditionally writes `tag`/`pretty_name` (a tagless 2-arg-component
   tuple gains both) and drops component `#access` (Optional). They are
   therefore **F-P5 seam-attribute cases deferred to Phase 4.5**, not clean
   4.3 internal migrations. The **Callable** builder (an empty-argument
   `code_typet`) is *not* raw byte-identical either (back-migration adds an
   `arguments` sub the source lacks) but **was migrated** — it is
   **IREP2-equivalent** (`migrate_type` canonicalises the empty `arguments`,
   so the type symex consumes matches the legacy form) and **GOTO-identical**
   (goto-convert normalises the leftover sub), §15.7. The lossless struct case
   is the `complex` struct (3-arg components + `tag` already present, §15.7).

### Phase 4.4 — Migrate **expression** construction to `expr2tc` (internal) — **REMAINING (the bulk; see the per-file guide in §7.2)**
7. Rewrite the expression converters to build `expr2tc` via typed factories
   instead of `symbol_expr`/`typecast_exprt`/`member_exprt`/`index_exprt`/
   `address_of_exprt`/`dereference_exprt`/`constant_exprt`/
   `side_effect_expr_function_callt`. Split by file/handler — one commit
   each, roughly in dependency order: `converter_symbols.cpp` (symbol refs)
   → `converter_expr.cpp` → `converter_unop.cpp`/`converter_binop.cpp`/
   `converter_compare.cpp` → `converter_funcall.cpp` → the big handlers
   (`python_list.cpp`, `python_dict_handler.cpp`, `python_set.cpp`,
   `python_math.cpp`, `numpy_call_expr.cpp`, `string/*`). Back-migrate to
   `exprt` only where the result is stitched into a legacy `codet` body
   (Phase 4.5). **This is the bulk of the work and the highest churn-
   collision risk** — keep commits small and land them fast.

### Phase 4.5 — The body boundary (what stays legacy, made explicit) — **REMAINING**
8. Localize the legacy hand-off: statement assembly (`converter_stmt.cpp`)
   keeps emitting structured legacy `codet` (P1), but each statement's
   *operands* are now IREP2, lowered through a single documented
   `migrate_expr_back` helper at the point the statement is built.
   `complex`/`#cpp_type`-typed and struct-component (`#member_name`)
   operands re-attach their seam attributes here (F-P5). Document this as
   the durable Python frontend boundary — the analogue of Part I's
   `migrate_expr`-at-goto-convert seam and Part III §3.5.

### Phase 4.6 — Tighten & census — **REMAINING**
9. Retire now-dead legacy includes/helpers; snapshot the Python IREP2-share
   census and record it as Part IV's outcome section (mirror Part II §11 /
   Part III §14). Reassess whether to proceed past the seam or, as Solidity
   did, declare the encapsulated + IREP2-internal converter the stopping
   point.

### Out of scope — separate tracking issues (the deep pins, shared with Part III)
- **IREP2-native goto-convert input** (structured CF code kinds + IREP2
  `goto_convert_rec`). Removes P1. Affects every frontend.
- **IREP2-native `clang_cpp_adjust`** (or a Python-specific adjust that does
  not round-trip legacy and reads `#cpp_type` natively). Removes P2.
- **IREP2-native C printer for Python counterexamples** (Part II Phase 2.7).
  Witness/`test.desc` text-parity bar.

## 7.1 Acceptance criteria per phase

| Phase | Done when |
|---|---|
| 4.0 | Golden verdict+text set captured on asserts build over the stratified subset; `ast2json` baseline recorded; equivalence harness compiles and passes against legacy builders. |
| 4.1 | `grep -rE '\.(set\|get)\("#' src/python-frontend` returns only the typed-accessor bodies; each attribute has one repoint seam; suite verdict+text unchanged on both solvers. |
| 4.2 | New factories/helpers have round-trip unit tests; **no converter call site changed**; suite unchanged. |
| 4.3 | Type builders return `type2tc`; back-migration confined to symbol store + seam helper; `python_int_typet` width identical (unit-pinned); suite verdict+text unchanged on both solvers; asserts-build cross-check silent. |
| 4.4 | Expression converters build `expr2tc`; legacy `symbol_expr`/`*_exprt`/`side_effect_expr_function_callt` count in `src/python-frontend` strictly decreasing PR-over-PR; suite unchanged. |
| 4.5 | Exactly one documented `migrate_expr_back` hand-off feeds legacy `codet` bodies; `#cpp_type`/`#member_name` re-attached at the seam; structured CF still lowered by `goto_convert`; suite unchanged. |
| 4.6 | Census recorded; dead legacy retired with C-Dead discharge; suite unchanged; go/stop decision documented. |

The bar for **every** phase: identical pass/fail set and identical matched
output text under **both Bitwuzla and Z3**, on an asserts-enabled build,
over the stratified `regression/python` corpus (full suite at phase
boundaries, respecting the 5-minute cap by narrowing per commit).

## 7.2 Phase 4.4 execution guide (per-file, ordered) — the remaining bulk

Phase 4.4 is the largest, highest-risk phase (RP13) and the next to execute.
This section is the step-by-step recipe an engineer new to the frontend can
follow. **Prerequisite — SATISFIED (#4997, §0.1):** Phase 4.2's expression-side
factories `symbol_expr2tc(const symbolt&)` and `side_effect_function_call2tc(...)`
(`src/util/migrate.{h,cpp}`) have landed dead-but-tested; Phase 4.4 *wires*
them — it does not build them. Each numbered file below is **one or more
reviewable commits**; apply and gate (§10) one at a time; never batch; rebase
daily (RP1).

### 7.2.1 The idiom → factory substitution table (the mechanical core)

For every expression-construction idiom, the IREP2 replacement and the
gotcha. This is the lookup an engineer applies at each call site.

| Legacy idiom | IREP2 replacement | Gotcha / verification |
|---|---|---|
| `symbol_expr(sym)` (842 sites) | `symbol_expr2tc(sym)` (landed #4997, `migrate.h:61`) | The helper already reads the IREP2 source of truth via `migrate_symbol_type(sym)` (not a back-migrate) and is proven `==` to `migrate_expr(symbol_expr(sym))`. RP9 overload hazard: `symbol_expr` (legacy) vs `symbol_expr2tc` (IREP2) differ by one token and both compile — grep each converted site. |
| `typecast_exprt(e, t)` (131) | `typecast2tc(t2, e2)` | Argument order flips (type first in IREP2). Width/signedness must match exactly (U3/RP4). |
| `member_exprt(base, name, t)` (121) | `member2tc(t2, base2, name)` | `name` is the component `irep_idt`; for `#member_name`-tagged structs re-attach at the 4.5 seam (RP3). |
| `index_exprt(arr, idx, t)` (22) | `index2tc(t2, arr2, idx2)` | The `index < l->size` bounds guard (`python_list.cpp`) is the **one** pretty-print gate (Q-P1/§15.6) — keep its rendered text byte-identical. |
| `address_of_exprt(e)` (87) | `address_of2tc(subtype, e2)` | Pointer subtype must match `pointer_type2tc(e2->type)`. |
| `dereference_exprt(p, t)` (37) | `dereference2tc(t2, p2)` | — |
| `constant_exprt` int/float/bool (42) | `constant_int2tc` / `constant_floatbv2tc` / `gen_boolean` | `#cformat` is set by the legacy `constant_exprt` ctor (std_expr.h:1095); preserve via the standard mechanism (Q-P2). Use `from_double` (§6 gap) for floats. |
| `side_effect_expr_function_callt` (111) | `side_effect_function_call2tc(return_type, function, arguments)` (landed #4997, `migrate.h:67`) | The §6 wrapper; do not hand-roll `sideeffect2tc` — the helper fixes the nil-size / empty-alloctype canonical form (`migrate.cpp:446-450`). Function/args/return-type all required. |
| `e.operands()[i]` / `e.op0()` (167) | typed field of the specific `*2t` kind | **The hard part.** Each access must be hand-translated to the named field (`.value`, `.ptr_obj`, `.side_1`…). A mis-indexed operand silently builds the wrong node (RP13). |
| `e.id() == "constant"` (146) | `is_constant_int2t(e)` etc. | Compile-time predicate — a missed branch is now a *compile* error, which is the soundness win (Part I §"Why this mattered"). |
| `e.find("...")` raw sub-irep (167) | typed field, or audit per site | No generic analogue; some have none — audit each. |
| `code_*t` statement bodies | **stay legacy `codet` (P1)** — only their operands move | Phase 4.5 owns the hand-off; do not migrate the statement shells. |

### 7.2.2 Converter file inventory and migration order

Ordered lowest-entanglement-first so the mechanical idiom is exercised on
easy files before the dense OM handlers. Counts are this-revision snapshots
(re-run §2). "Surgery" = `move/copy_to_operands` + `.operands()` + `.id()` +
`.find(`.

| Order | File | ~LOC | Role | Surgery density | Notes |
|---|---|---:|---|---|---|
| 1 | `converter/converter_unop.cpp` | 166 | unary ops | low (9 `symbol_expr`) | **START HERE — the genuine warm-up.** Holds statement-free `not`-on-{string,dict,list} truthiness subtrees that lower cleanly to one back-migrate. The first commit landed the `not`-on-string subtree (`strlen(base)==0`) via `symbol_expr2tc` + `side_effect_function_call2tc` + `equality2tc`; dict/list cases (which emit `code_*t` into `current_block`) follow. |
| 2 | `converter/converter_expr.cpp` | 1451 | central dispatch: Name/Attribute/Subscript/Constant/Call | low–med (8) | The hub; member/index/typecast/const/deref live here. |
| ~~—~~ | ~~`converter/converter_symbols.cpp`~~ | 385 | **NOT a wiring target** | n/a (**0** `symbol_expr`) | **Correction (earlier drafts listed this as "file 1, migrate symbol-ref construction first").** Re-audit shows it does symbol-table *lookup* (`find_symbol`/`find_imported_symbol`/… → `symbolt*`) and the *write boundary* (`update_symbol`/`create_tmp_symbol`, `set_type`/`set_value` at 36/40/382). The write boundary stays legacy (F-P1); there is **no `symbol_expr` construction here to migrate** (`grep -c 'symbol_expr(' = 0`). Skip it. |
| 3 | `converter/converter_compare.cpp` | 632 | comparisons (incl. `in`/`not in`) | low | — |
| 4 | `converter/converter_binop.cpp` | 1419 | binary ops; heavy `typecast` coercion (20) | med | Coercion width/signedness is U3/RP4-sensitive. |
| 5 | `converter/converter_funcall.cpp` | 1185 | call dispatch, builtins/lambda/method routing | med | Feeds the OM handlers; migrate before them. |
| 6 | `converter/converter_class.cpp` | 786 | class/inheritance/method extraction | med | `#member_name` struct components surface here (RP3). |
| 7 | `converter/converter_dunder.cpp` | 273 | dunder dispatch | low | — |
| 8 | `complex_handler.cpp` | — | complex arithmetic (23 `member_exprt`) | med | Pair with the `complex` struct (RP7, Q-P5 harness). |
| 9 | `python_math.cpp` | 1404 | math model calls | med (44) | Simpler "wrap call" pattern. |
| 10 | `numpy_call_expr.cpp` | 1037 | numpy model | low (8) | Sets `#cformat` (Q-P2/Q-P3) — preserve. |
| 11 | `function_call/str_conv.cpp` + `string/` | — | string methods; already has the `migrate_*` round-trip (F-P9, 559/573) | med (53 in string_method_handler) | String **literals** stay legacy at the seam (Q-P4) — do not port `mb_value`. |
| 12 | `python_dict_handler.cpp` | 3302 | dict model | **high (78)** | OM handler; run the dict regression stratum every commit (RP5). |
| 13 | `python_set.cpp` | 1240 | set model; reads `mb_value()` at 107 (on a legacy node it just built — leave legacy) | **high (85)** | OM handler. |
| 14 | `python_list.cpp` | 4905 | list model; the bounds guard (Q-P1) | **highest (132)** | **Most entangled file in the tree — last, most commits, most care.** Split by method family (`build_insert_list_call`, `build_split_list`, `handle_index_access`, slice/repetition). |

(`converter_symbols.cpp` is intentionally absent — see the struck row above.)

**Why this order.** Files 1–7 are the converter core: low surgery density,
high reuse, so the idiom table is debugged there. Files 8–11 are
medium-density specialists. Files 12–14 are the operational-model handlers
that hold ~80 % of the raw operand surgery (RP13) and are themselves
converter *input* (§3.1, RP5) — a regression there breaks every program
importing the model, so they go last, in the smallest commits, each gated by
their own regression stratum on every commit.

**Two converter files are handled outside the 1–15 expression sequence — by
design, not omission** (re-verify counts with §2; figures are 2026-06-02):

| File | ~LOC | Surgery | Where it belongs | Why not in the 1–15 list |
|---|---:|---|---|---|
| `converter/converter_funcdef.cpp` | 1235 | **low** (4 operand + 4 `.id()`) | fold into the **core pass** (between files 6 and 7) | It builds function-definition *symbols* and the body `code_blockt` shell. The shell stays legacy (P1); only its operand expressions and the synthesized `code_type2t` parameter names (RP6) move. Low surgery → safe early warm-up; it was missing from the original table — add it as an early core commit. |
| `converter/converter_stmt.cpp` | 3624 | **high** (75 operand + 23 `.id()` + 9 `.find(`) | **Phase 4.5**, not 4.4 | This is the statement-assembly file and the **home of the single documented `migrate_expr_back` body seam** (Phase 4.5). Its `code_*t` statement *shells* (`while`/`ifthenelse`/`block`/`decl`/`assign`/`return`) stay legacy by design (P1, Q-P6); only the *operands* it stitches in are IREP2. So it is touched in 4.4 only to the extent of feeding it IREP2 operands, and *finalized* in 4.5 when the hand-off is localized. Do **not** migrate its statement shells in 4.4. |

`converter/converter_types.cpp` (819 LOC, 5+2+2 surgery) is the JSON-class→
`typet` bridge (`get_type_from_annotation`, §3.2 investigation note); it is
**Phase 4.3** type-side work, mostly landed, and any unhandled annotation
string it encounters is a pre-existing SM-class gap to preserve, not fix (§14).

### 7.2.3 The per-commit recipe

> **The unit of migration is a *subtree*, not a *site* (learned wiring commit
> #1).** A single leaf swap — replacing one `symbol_expr(sym)` with
> `symbol_expr2tc(sym)` in isolation — buys **nothing**: its consumer is still a
> legacy `exprt`/`codet` constructor, so the result must be `migrate_expr_back`'d
> immediately, reproducing the identical node with an added round-trip. Net
> IREP2 surface gained: zero. The migration only pays off when a **contiguous
> expression subtree, from its leaves up to a statement/return boundary**, is
> built in `expr2tc` and back-migrated **once** at that boundary. So pick the
> work-unit by *subtree*: the smallest self-contained expression a function
> *returns* or *assigns into a statement operand*, with one back-migrate at its
> root. Statement-free subtrees (e.g. a function that returns a comparison
> without emitting into `current_block`) are the cleanest starting units —
> wiring commit #1 used exactly such a subtree (`converter_unop.cpp`'s
> `not`-on-string `strlen(base)==0`, file 1 of §7.2.2).

For each file (or each method family within `python_list.cpp`):

1. **Read the whole function** before touching it. Note every `.operands()`,
   `.id()`, `.find(` — these are the RP13 sites that do not map mechanically.
   Identify the **subtree boundaries**: where does an expression get returned or
   stitched into a `code_*t` operand? Those roots are your back-migrate points.
2. **Translate construction bottom-up**: build leaf `expr2tc` first, compose
   upward with typed factories (idiom table §7.2.1). Migrate any *legacy* input
   the function receives (e.g. a helper that still returns `exprt`) forward with
   `migrate_expr` at the point it enters the subtree.
3. **At the statement-assembly / return point, back-migrate once**
   (`migrate_expr_back`) so the legacy `codet` body or the legacy-typed return
   still receives a legacy `exprt` (Phase 4.5 will localize this into one
   documented helper; until then it is an explicit call at the subtree root). Do
   **not** migrate the `code_*t` shell. **Do not** swap a leaf whose subtree root
   you are not also converting in the same commit — that adds a back-migrate for
   no gain (the subtree-not-site rule above).
4. **Build asserts-enabled** so `migrate.cpp`'s symbol round-trip cross-check
   runs live over the corpus (§10.3).
5. **Gate** (§10): verdict set identical + matched-text identical, **both
   Bitwuzla and Z3**, on the file's regression stratum every commit; the
   bounds-guard test (`regression/python/bounds-checking_fail`, Q-P1) and the
   `complex_type_test`/round-trip unit tests must stay green.
6. **Census**: the legacy-builder count for that file must strictly decrease;
   record it (§12 primary metric).

### 7.2.4 The RP13 sub-protocol (operand surgery — the part that is *not* mechanical)

The 582 `move/copy_to_operands` + 167 `.operands()` sites are where wrong
verdicts hide. For each:

- **Identify the target `*2t` kind** the legacy node would have become at
  `migrate_expr`. Read `migrate.cpp`'s forward switch for that `id()` to see
  which fields it populates from which operand index.
- **Replace positional operand access with the named field.** E.g. a
  `code_function_callt` built via `copy_to_operands(func, arg0, arg1)` becomes
  `code_function_call2tc(ret, func2, {arg0_2, arg1_2})` — and any later
  `call.operands()[2]` becomes `call.operands[1]` on the typed `arguments`
  vector, **not** a raw irep index.
- **Pair every `.id()==` branch with its `is_*2t` predicate.** A dropped
  branch is now a compile error (the win); a *wrong* predicate is a silent
  bug — review each against the legacy string.
- **Per-site review is mandatory** (RP9: name-identical overloads). `-Werror`
  on. Two reviewers on `python_list.cpp`.

### 7.2.5 Worked operand-surgery examples (the non-mechanical core, verified)

§7.2.4 states the protocol; this section *executes* it once per major `*2t`
kind so an engineer who has never opened `irep2_expr.h` has a concrete
before/after to copy. **Every field name below is verified against the cited
`irep2_expr.h` line at this revision — re-confirm before relying on it, the
header moves.** The factory call order follows each class's primary
constructor: **type first**, then sources/operands.

Field map for the kinds the Python converter builds most (from
`src/irep2/irep2_expr.h`):

| Kind | Class line | Constructor | Fields (in order) |
|---|---|---|---|
| `symbol2t` | 558 | `symbol2tc(type, name)` | `irep_idt thename` (l.563) — the identifier |
| `typecast2t` | 662 | `typecast2tc(type, from)` | `expr2tc from` (l.665) |
| `member2t` | 1485 | `member2tc(type, source, memb)` | `expr2tc source_value` (l.1488), `irep_idt member` (l.1489) |
| `index2t` | 1563 | `index2tc(type, source, idx)` | `expr2tc source_value`, `expr2tc index` (l.1566-67) |
| `dereference2t` | — | `dereference2tc(type, ptr)` | `expr2tc value` |
| `address_of2t` | — | `address_of2tc(ptr_subtype, obj)` | `expr2tc ptr_obj` |
| `sideeffect2t` (call) | 1749 | use `side_effect_function_call2tc(ret, fn, args)` | `kind == allockind::function_call` |

**Example A — `member_exprt` → `member2t`** (the 121-site idiom; `complex`
real/imag access in `complex_handler.cpp`, struct components in
`converter_class.cpp`). Legacy:

```cpp
// base is an exprt of struct type; field "real" has type double
member_exprt m(base, "real", double_type());
```

IREP2 (build the base `expr2tc` first, then compose):

```cpp
// base2 : expr2tc of struct type (already migrated upstream)
expr2tc m2 = member2tc(double_type2(), base2, "real");
```

Gotcha: `member2t`'s constructor `assert`s `source->type` is a struct/union
(`irep2_expr.h:1499`); an asserts build catches a mis-typed base immediately
— which is the soundness win. For a `#member_name`-tagged component the *tag*
is re-attached at the 4.5 seam, not on the IREP2 node (RP3/F-P5).

**Example B — positional `.operands()` read → named field** (the dangerous
167-site class). A legacy site that pulls operand 0 of a typecast:

```cpp
exprt inner = cast_expr.operands()[0];   // or cast_expr.op0()
```

becomes a *named* field access on the typed node — never a positional index:

```cpp
const expr2tc &inner = to_typecast2t(cast2).from;   // typecast2t::from, l.665
```

The failure mode this kills: `operands()[0]` on a node whose operand layout you
misremember silently reads the wrong child; `to_typecast2t(x).from` cannot —
the field is named and the cast asserts the kind.

**Example C — `side_effect_expr_function_callt` → the landed helper** (111
sites; the OM-call backbone). Legacy:

```cpp
side_effect_expr_function_callt call;
call.function() = symbol_expr(model_fn_sym);
call.arguments().push_back(arg0);
call.arguments().push_back(arg1);
call.type() = ret_typet;
```

IREP2 (Phase 4.2 helper — do **not** hand-roll `sideeffect2tc`):

```cpp
expr2tc call2 = side_effect_function_call2tc(
  ret_type2,                          // type2tc
  symbol_expr2tc(model_fn_sym),       // callee, also the landed helper
  {arg0_2, arg1_2});                  // std::vector<expr2tc>
```

This is proven `==` to `migrate_expr` of the legacy form
(`unit/util/migrate.test.cpp:246`), so the symex input is byte-stable.

**Example D — `.id()` string branch → compile-time predicate** (146 sites).
Legacy dispatch:

```cpp
if (e.id() == "constant")        { ... }
else if (e.id() == "symbol")     { ... }
```

becomes:

```cpp
if (is_constant_int2t(e2))       { ... }   // or is_constant_*2t per the type
else if (is_symbol2t(e2))        { ... }
```

The win (Part I "Why this mattered"): a branch you *forget* is now a compile
error under `-Werror`, not a silently-untaken path at symex. The hazard (RP9):
a branch you translate to the **wrong** predicate compiles and is wrong — so
each `.id()==` string must be matched to its predicate against `migrate.cpp`'s
forward switch (`src/util/migrate.cpp`, the `expr.id() == ...` ladder from
l.711), which is the authority on which `id()` string lowers to which `*2t`.

**The mandatory cross-check after every example.** Build asserts-enabled and
let `migrate.cpp`'s symbol round-trip run over the corpus (§10.3): if your
hand-built `expr2tc` is *not* what `migrate_expr` would have produced from the
legacy node, the round-trip cross-check or a constructor assert fires before
symex — which is exactly why the migration is done with asserts on.

## 8. Dependencies and prerequisites

- **4.1 is independent and unblocking** — do it first; it is the milestone
  even if the project stops there (Part III §14 precedent).
- **4.2 → DONE (#4997, §0.1).** The factories that gated 4.3/4.4 are landed
  and tested; this dependency is discharged. (Kept here for the record: the
  V-track lesson held — dead-but-tested infrastructure shipped as its own PR.)
- **4.3 blocks 4.4** (expressions carry their `type2tc`; types must be
  buildable first). Elementary/float/array/pointer/Callable types are landed
  (§0); the tuple/optional struct builders are deferred to 4.5 (§15.7), so a
  4.4 expression whose *type* is a tuple/optional struct must back-migrate that
  type at the seam until 4.5 lands — not a blocker, but flag such sites.
- **4.4's string/`bytes` work depends on Q-P4** (`mb_value`).
- **4.5 depends on P1 staying pinned.** If the out-of-scope IREP2
  goto-convert ever lands, 4.5's hand-off is where it connects.
- The whole plan depends on the **goto-binary on-disk format not changing**
  (Part I, B4), on the **JSON AST contract** (§11), and on the **`#cpp_type`
  / `#member_name` shared-pass reads remaining satisfied** at the seam
  (F-P5).

## 9. Risk register (Python-specific; extends Part II §7 / Part III §9)

| # | Area | Risk | Sev | Mitigation |
|---|---|---|---|---|
| RP1 | **process** | The Python frontend is the highest-churn tree in the repo; a multi-week migration branch will conflict continuously against a moving target. | **high** | Small, fast-landing commits; rebase daily; never let a phase branch live > a few days; coordinate a soft freeze on the touched files per phase. |
| RP2 | soundness | `#cpp_type` is read by the **shared** `clang_cpp_adjust` (F-P5); an IREP2-internal type built without re-attaching it at the seam silently changes adjust's behaviour for float/char symbols → wrong cast/width. | **critical** | Seam helper re-attaches `#cpp_type`; unit test round-trips a `#cpp_type`-tagged type and asserts the legacy node still carries it; `convert_float_literal`/`type_handler:564` paths covered. |
| RP3 | soundness | `#member_name` (shared clang-cpp struct-component convention) lost at the seam → constructor namespace lookup / mangling misfires (Part III Q-S2). | **critical** | Same as RP2: re-attach at the seam; never migrate it off the IR; covered by class/struct regression tests. |
| RP4 | soundness | `python_int_typet` width logic (64-bit default vs 512-bit `--ir`; F-P7/SM1) altered during migration flips overflow/wrap verdicts. | high | Pin the width in a unit test **before** Phase 4.3; assert byte-identical `typet` pre/post; keep `--overflow-check` Python tests green on both modes. |
| RP5 | soundness | Operational-model `.py` files (§3.1) are themselves converter input; a converter regression breaks *every* program importing that model, not just one test. | high | The model `.py` corpus is in the regression set; run `numpy`/`dataclasses`/`decimal`/`collections` tests every commit, not just at phase close. |
| RP6 | correctness | `code_type2t` arity assertion (`args.size()==names.size()`, `irep2_type.h:240`) — Python builds function types without parameter names in places. | med | Supply synthesized names in 4.3; asserts build catches it. |
| RP7 | soundness | `complex` is modelled as a `struct` with `#member_name`-tagged components and a cached static `struct_typet` (`type_handler.h:18-34`); IREP2-building it must reproduce the exact tag/component layout or `migrate_type` produces a different `struct_type2t`. | high | Unit-pin the `complex` struct layout; the existing `unit/python-frontend/complex_type_test.cpp` is the oracle — extend it for the IREP2 path. |
| RP8 | soundness | `str`/`bytes` decode (`mb_value`, UTF-8/wide/endianness) lives only on the legacy class (Part II R10) and is the trickiest logic; an IREP2 port can diverge. | med | Carry the decoder **verbatim**; property-test against CPython (AGENTS.md "Hypothesis tests"); or keep literal lowering legacy at the seam (Q-P4). |
| RP9 | soundness | Name-identical legacy/IREP2 overloads (`gen_zero`, `member`, `symbol_expr` vs `symbol2tc`) — both compile, wrong one binds (Part II R12). | med | Per-site review; `-Werror`; unit equivalence tests. |
| RP10 | compatibility | Counterexample text: Python `test.desc` regexes match function names, line numbers, sometimes coverage counts. A construction change that alters symbol naming or pretty-printed types breaks them. | high | §10 captures matched text, not just verdict; keep the legacy C printer (Part II 2.7 out of scope). |
| RP11 | scope-creep | The "real" migration tempts touching P1 (goto-convert) / P2 (adjust); doing so blows blast radius across all frontends. | med | P1/P2 are separate tracking issues; Part IV stops at the body seam (4.5). |
| RP12 | environment | Python tests require `ast2json` and spawn an external `python3`; a stale/absent interpreter masquerades as a verdict regression. | med | Pin the baseline `ast2json`/interpreter; run inside the project venv (AGENTS.md); treat parse failures as infra, not migration, regressions. |
| RP13 | correctness/effort | **Heavy raw irep operand surgery** (§15 census: 576 `move/copy_to_operands`, 166 `.operands()`, 142 `.id()`, 161 `.find(`), concentrated in `python_list`/`python_set`/`python_dict_handler`/`converter_stmt`. Each `.operands()[i]`/`.id()==…`/`.find("…")` site must be hand-translated to typed `*2t` field access; a mis-indexed operand or a missed `.id()` branch silently builds the wrong node. | **high** | Phase 4.4 split per handler, one commit each; per-site review (RP9 overload hazard compounds here); the §10 verdict+text gate after every handler; the OM-handler files (`python_list` etc.) are the highest-risk and run their regression stratum every commit (RP5). |

## 10. Validation, regression & equivalence strategy

Reuse the Part II universal gate, specialized for Python. The suite is
`regression/python` (**3558 test dirs**) plus `regression/python-coverage`
and `regression/python-intensive`; label `python` (registered when
`-DENABLE_PYTHON_FRONTEND=On`). Per AGENTS.md, Python tests need `ast2json`
in the invoked `python3`, and each test spawns the external parser.

**Per-phase gate (all must hold, both Bitwuzla and Z3):**
1. **Verdict set identical** to the 4.0 baseline. Verdicts are immune to
   the model-naming nondeterminism that makes `diff_goto_baseline`
   unreliable across rebuilds (Part II §8.1), so this is the primary oracle.
   *Do not rely on `diff_goto_baseline` here* — the Python operational-model
   bake is one of the documented nondeterministic loci (§8.1 names
   `esbmc-python-astgen-<hex>` temp paths explicitly).
2. **Matched-text identical.** Capture and diff the `test.desc`-matched
   lines (function names, line numbers, coverage counts), not just the
   verdict. Backstop for RP10.
3. **Asserts-enabled build** so the `migrate.cpp` symbol round-trip
   cross-check stays live across the corpus — the strongest evidence the
   converter's new output is still losslessly convertible.
4. **Affected unit suites** green, including the new
   `unit/python-frontend/` IREP2-construction equivalence tests and the
   extended `complex_type_test.cpp` (RP7).
5. **Run subset first, full last.** Stratified CORE subset on every commit;
   `python-intensive` / full 3558 at phase boundaries only. Respect the
   project 5-minute full-suite cap — narrow scope per commit (the full suite
   far exceeds it; pick a representative stratum covering int/float/str/
   list/dict/set/class/complex/numpy/overflow). Clean
   `/tmp/esbmc-headers-*` after runs (AGENTS.md "/tmp disk space").

**Phase-4.3-specific equivalence test (the load-bearing one):** for a
representative corpus of annotated JSON type nodes, assert
`migrate_type_back(build_type2tc(node)) == legacy_build_typet(node)`
**including** the `#cpp_type`/`#member_name` re-attachment, for every node.
This pins RP2/RP3/RP4/RP7 by construction.

## 11. Backward-compatibility considerations

- **Goto-binary on-disk format: unchanged** (Part I, B4 / RETAIN_BOUNDARY).
  Old `.goto` binaries must still load; the Python path produces the same
  serialized IR.
- **JSON AST contract: unchanged.** The frontend still spawns
  `parser/__main__.py` (ast2json) and consumes the same JSON shape; the
  `python_annotation` pass is untouched. Do **not** change the parser as
  part of this migration (it would conflate ast2json drift with refactor
  effects — the analogue of Part III's "do not regenerate `.solast`").
- **CLI surface unchanged:** `--python`, `--function`, `--ir` (bignum),
  `--deadlock-check`, `--parse-tree-only` (`python_language.cpp`).
- **Operational models (C `c2goto/library/python/*` and Python
  `models/*.py`): unchanged.** The converter still emits the same calls into
  them; model-selection logic must pick the same model (this is what the
  verdict gate enforces).
- **`#cpp_type` / `#member_name` seam contract: preserved** (F-P5/RP2/RP3) —
  the legacy node handed to `clang_cpp_adjust` still carries them.
- **Counterexample / diagnostic text: preserved** where `test.desc` matches
  it (RP10). The legacy C printer stays (Part II 2.7 out of scope).

## 12. Success metrics and exit criteria

- **Primary (this plan):** the legacy-builder census in
  `src/python-frontend` (`symbol_expr`, `*_exprt`,
  `side_effect_expr_function_callt`, `constant_exprt`, non-CF `code_*t`
  *operands*) strictly decreasing PR-over-PR; the Python IREP2-share rising
  from <1 % (Part I surface table) — measured by the §2 census.
- **Secondary (soundness milestone, Phase 4.1):** raw
  `.set("#…")`/`.get("#…")` access outside the typed accessors reaches
  empty; the three attributes sit behind one repoint seam each.
- **Exit (this plan):** the converter builds types and statement-operand
  expressions in IREP2 internally, and hands off to
  `clang_cpp_adjust`/`goto_convert` through **one documented legacy seam
  (4.5)** that re-attaches `#cpp_type`/`#member_name`. Function-body
  control-flow lowering (P1), the shared `clang_cpp_adjust` pass (P2), the
  bignum int approximation (SM1, deliberately unchanged), and the Python C
  printer remain legacy under their own tracking issues. Verdict+text parity
  holds across the stratified suite on both solvers.

The bar is Part I's bar: **behaviour-preserving at every step, proven, not
assumed.** As Part III found for Solidity, the honest end state may be the
encapsulated, IREP2-internal converter with a legacy seam — that is genuine
forward progress, not a half-migration, and is preferable to forcing a node
across the seam that drops a shared-pass-read attribute.

## 13. Open questions (resolve before the cited commit)

- **Q-P1 — RESOLVED (§15.6): the symbol-naming gate is real but narrow.**
  Of 3503 expected-output lines across 3660 Python `test.desc` files,
  **exactly one** regex matches pretty-printed IR expression text
  (`bounds-checking_fail`: `index < l->size`, `--ir`); **none** match
  concrete counterexample variable values. Plus a coverage-count family (39
  lines) and a frontend-diagnostic family (46 lines) that must be held
  invariant. So Phase 4.4's RP10 bar is cheap to gate. See §15.6.
- **Q-P2 — RESOLVED (§15): not dead, do not delete.** `#cformat` is shared
  irep infrastructure (`irept::a_cformat`, `irep.cpp:507`;
  `constant_exprt`'s ctor sets it, `std_expr.h:1095-1099`) read by the
  pretty-printers for constant formatting. Python sets it (set-only in the
  frontend) to feed downstream C text. Phase 4.1 encapsulates it via the
  existing standard mechanism; it stays on the legacy seam node.
- **Q-P3 — RESOLVED (§15): all three are externally read; all stay on the
  seam.** `#cpp_type` is read by **three** consumers
  (`clang_cpp_adjust_expr.cpp:464`, `cpp_expr2string.cpp:138-140`,
  `goto2c/expr2c.cpp:169-174`); `#member_name` by the shared clang-cpp pass;
  `#cformat` is shared infra (Q-P2). None can move off the IR; this hardens
  the F-P5 / 4.5 seam contract. (No reads found in `goto-symex`; the
  `goto-programs`/`util`/`goto2c` readers are the pretty-printer + emitter
  paths, which is why the legacy C printer staying is a hard dependency.)
- **Q-P4 — RESOLVED: string-literal lowering stays legacy at the seam; do
  not port `mb_value`.** The Python frontend builds **legacy
  `string_constantt`** for `str`/`bytes` literals (`converter_stmt.cpp:3031`,
  `exception_utils.cpp:26`, `python_exception_handler.cpp:142-153`) and reads
  `mb_value()` in **exactly one** place, on a legacy node it just built
  (`python_set.cpp:107`). The `string2array`/`array2string` decode paths have
  **no** Python-frontend callers (§4: their only callers are
  `c_typecast.cpp:750` / `io.cpp:233`). Since Phase 4.5 keeps the statement
  body legacy `codet` anyway, string literals naturally live at that seam and
  `mb_value()` keeps being read off the legacy class — so the R10/Part-II-2.6
  verbatim port is **not** needed for the Python migration and is avoided
  (RP8 sidestepped). Re-open only if a later phase migrates string-literal
  *construction* itself past the seam.
- **Q-P5 — RESOLVED (Phase 4.0 harness, #4992): the `complex` struct
  round-trips faithfully.** `unit/python-frontend/irep2_type_roundtrip_test.cpp`
  pins that `get_complex_struct_type()` survives `migrate_type` /
  `migrate_type_back`: `is_complex_type` still holds, the tag stays `complex`,
  and both components round-trip in order (`real`, `imag`, each
  `double_type()`). The same harness records the F-P5 corollary — the
  round-trip **drops `#cpp_type`** — confirming RP7's re-attach-at-the-seam
  requirement for Phase 4.5. (was: blocks 4.3 complex path / RP7)
- **Q-P6 — RESOLVED (§15): For→While in the Python preprocessor; While is a
  generic `codet("while")`.** `for` is rewritten to `while` (with for-else
  lowering, `break` rewriting, and known-list-literal unrolling) at the
  **Python preprocessor layer** (`preprocessor/loop_mixin.py:211
  visit_For`), so no `For`/`code_fort` ever reaches the C++ converter.
  `while` is emitted as a legacy `codet` via `set_statement("while")` with a
  `code_blockt` body (`converter_stmt.cpp:2362-2364, 2734-2735`). So the
  Phase 4.5 seam produces structured `codet("while"/"ifthenelse")` +
  `code_blockt` shells whose *operands* are the IREP2-lowered expressions —
  confirming P1 and fixing the exact body shape the seam emits.

## 14. Semantic-mismatch & potential-unsoundness catalogue (Python-specific)

The user asked explicitly for the places where semantic mismatch,
unsupported features, or verification unsoundness could be **introduced** by
this migration, versus those that **pre-exist** and must be preserved
unchanged. The distinction is critical: the migration's job is to preserve,
not to fix — fixing a semantic gap during an IR migration conflates two
effects and violates the behaviour-preserving rule.

**Pre-existing (must be preserved byte-for-byte, NOT fixed here):**
- **SM1 — Python `int` is not arbitrary-precision.** Default lowering is
  64-bit `long_long_int_type`; `--ir` widens to 512-bit `signedbv`
  (`type_handler.cpp:43-51,471` FIXME). Genuine `int` overflow semantics are
  approximated. Preserve the exact width selection (F-P7/RP4).
- **SM2 — `str` is a C `char` array/pointer**, not a Unicode-aware Python
  `str`; multibyte decode via `mb_value` (RP8). Preserve the decoder
  verbatim.
- **SM3 — Operational-model fidelity.** `list`/`dict`/`set`/`numpy` semantics
  are whatever the C and `.py` models implement; the migration must call the
  *same* model with the *same* arguments (RP5).

**Could be INTRODUCED by this migration (the watch-list):**
- **U1 — Dropped seam attribute (RP2/RP3).** `#cpp_type`/`#member_name` not
  re-attached at the 4.5 seam → shared `clang_cpp_adjust` takes a different
  path silently. Highest-severity new hazard; the §10 round-trip test pins
  it.
- **U2 — `complex` struct layout drift (RP7).** An IREP2-built `complex`
  whose component order/tag differs from the cached `struct_typet` →
  `member`-access mismatch in the operational model.
- **U3 — Width/signedness drift in `type2tc` construction.** A
  `signedbv_type2tc` built with the wrong width (especially the 512-bit
  bignum path) flips overflow/comparison verdicts (RP4).
- **U4 — Overload misbinding (RP9).** A `gen_zero`/`member`/`symbol_expr`
  site silently binding the wrong (legacy vs IREP2) overload, both
  compiling, one wrong.
- **U5 — `code_type2t` arity assert (RP6)** turning a previously-accepted
  unnamed-parameter function type into a hard frontend abort — a
  *behaviour* change (crash vs verdict) even though it surfaces a latent
  inconsistency. Supply names; do not let it change which programs verify.

Each U-item maps to a §9 risk and a §10 gate; none may be left to symex to
catch. The governing rule stands: **for a verifier, a silently-dropped
attribute or a width drift is a wrong verdict, and a wrong verdict is the
only outcome this project exists to prevent.**

## 15. Investigation outcomes — Q-P2/Q-P3/Q-P6 resolved; a census correction

> This section records the resolution of the gating open questions that
> could be settled without writing code, **and a correction to a load-bearing
> census figure in the first draft of this part.** Honesty about the
> correction matters more than the draft looking clean: a verification plan
> that ships a wrong difficulty estimate is the same class of error the plan
> exists to prevent.

### 15.1 Census correction — operand-surgery surface (was "0", is large)

The first draft of §1 and the F-P3 table claimed the converters do
**no** raw irep operand surgery (`move_to_operands`/`copy_to_operands`/
`.operands()` → `0`). **That was a false negative**: the census command used
`grep -E 'a\|b'`, and under extended-regex `\|` matches a *literal pipe*, not
alternation, so it matched nothing. The same bug zeroed the
`symbol_exprt|symbol_expr` and `migrate_*` rows (those were caught and
corrected to 844 and "pre-seam" respectively in the first draft; the
operand-surgery row was not).

Corrected figures (re-run these — they use literal-string / proper-ERE
forms that do not trip the bug):

```sh
grep -rE 'move_to_operands|copy_to_operands' src/python-frontend | wc -l   # 576 (draft) → 582 (post-4.3)
grep -rF '.operands()' src/python-frontend | wc -l                         # 166 (draft) → 167
grep -rF '.id()'       src/python-frontend | wc -l                         # 142 (draft) → 146
grep -rF '.find('      src/python-frontend | wc -l                         # 161 (draft) → 167
grep -rE 'migrate_expr|migrate_type' src/python-frontend | wc -l           # 3 (draft) → 10 (post-4.3: +lower_to_seam seam in type_handler.cpp)
# concentration:
grep -rlE 'move_to_operands|copy_to_operands|\.operands\(\)' src/python-frontend \
  | xargs -I{} sh -c 'echo "$(grep -cE "move_to_operands|copy_to_operands|\.operands\(\)" {}) {}"' \
  | sort -rn | head
#   101 python_list.cpp / 83 python_set.cpp / 75 python_dict_handler.cpp /
#    74 converter_stmt.cpp / 53 string/string_method_handler.cpp / ...
```

**Impact on the plan.** Python is **more** irep-entangled than Solidity
(~124 `move_to_operands` total there), not less. Phase 4.4 is the dominant
effort and risk, the work is concentrated in the operational-model handlers
(`python_list`/`python_set`/`python_dict_handler`) and `converter_stmt`, and
the §1 bullet, F-P3 table, and new risk **RP13** have been updated to say so.
The earlier "mechanical factory-for-factory substitution" framing is
withdrawn. **Methodology note for future census work in this document:
never use `grep -E 'a\|b'`; use `grep -F`, `grep -e a -e b`, or
`grep -E '(a|b)'`.**

### 15.2 Q-P3 — every Python `#`-attribute is read by a shared/downstream pass

`grep -rn '#cpp_type\|#cformat\|#member_name' src --include=*.cpp --include=*.h`
restricted to readers outside `src/python-frontend`:

| Attribute | External readers | Verdict |
|---|---|---|
| `#cpp_type` | `clang_cpp_adjust_expr.cpp:464` (shared adjust); `cpp_expr2string.cpp:138-140` (C++ pretty-printer → counterexample text); `goto2c/expr2c.cpp:169-174` (goto→C emitter) | stays on legacy seam (3 consumers) |
| `#member_name` | `clang_cpp_adjust_code_gen.cpp`, `clang_cpp_convert.cpp` (shared clang-cpp) | stays on legacy seam (Part III Q-S2) |
| `#cformat` | `irep.cpp:507` (`a_cformat`), `std_expr.h:1095` (`constant_exprt` ctor) — shared irep infra read by pretty-printers | stays; standard mechanism |

No reads in `goto-symex`. This **hardens** F-P5 and the 4.5 seam contract:
the legacy node handed across the seam must still carry all three, and the
legacy C/C++ pretty-printer must stay (Part II 2.7 out of scope) because it
is the consumer of `#cpp_type`/`#cformat` for counterexample text.

### 15.3 Q-P2 — `#cformat` is shared infrastructure, not dead

Settled by 15.2: `#cformat` is `irept::a_cformat`, set by `constant_exprt`'s
constructor and read by the pretty-printers. Python's numpy fold sets it
(set-only in the frontend; `grep 'get("#cformat"' src/python-frontend` →
empty) to control downstream constant rendering. **Encapsulate via the
existing standard accessor in Phase 4.1; do not delete.**

### 15.4 Q-P6 — `for`→`while` in the Python preprocessor; `while` is generic `codet`

`preprocessor/loop_mixin.py:211` (`visit_For`) rewrites every `for` into a
`while` at the Python preprocessing layer — including for-else lowering,
`break`-flag rewriting, and unrolling loops over known list literals — so the
C++ converter never sees a `For` (hence zero `code_fort`). `while` is emitted
as a legacy `codet` (`set_statement("while")`, `code_blockt` body;
`converter_stmt.cpp:2362-2364, 2734-2735`). Phase 4.5's seam therefore emits
structured `codet("while"/"ifthenelse")` + `code_blockt` shells with
IREP2-lowered operands — P1 confirmed, body shape fixed.

### 15.5 All Part IV open questions now resolved

`Q-P1` resolved by audit (§15.6). `Q-P4` and `Q-P5`, which required building
something rather than just auditing, are now closed too:

- **`Q-P5`** (`complex` `struct_type2t` round-trip fidelity) is resolved by
  the **Phase 4.0 harness** (`unit/python-frontend/irep2_type_roundtrip_test.cpp`,
  #4992): the complex struct round-trips tag + components faithfully, and the
  harness additionally pins that the round-trip drops `#cpp_type` (F-P5 →
  re-attach at the 4.5 seam). See §13 Q-P5.
- **`Q-P4`** (`mb_value`) is resolved by decision: the Python frontend builds
  legacy `string_constantt` and reads `mb_value()` only on a legacy node
  (`python_set.cpp:107`); string-literal lowering stays legacy at the seam, so
  no verbatim R10 decoder port is needed. See §13 Q-P4.

### 15.6 Q-P1 — the Phase 4.4 symbol-naming / pretty-print gate is narrow

Audit of the matched-output regexes (line 4+) of **all 3660** Python
`test.desc` files (`regression/python` 3557, `python-intensive` 90,
`python-coverage` 13). Re-run:

```sh
find regression/python regression/python-intensive regression/python-coverage \
  -name test.desc | while read f; do awk 'NR>=4' "$f"; done > /tmp/qp1.txt
grep -c .                            /tmp/qp1.txt   # 3503 expected-output lines
grep -cE '^\^?VERIFICATION (SUCCESSFUL|FAILED)\$?$' /tmp/qp1.txt   # 3344 pure verdict
# the ONLY pretty-printed member/pointer expression in the whole suite:
grep -rhE '\->|\(\*|\.size' regression/python*/*/test.desc | grep -vE 'VERIFICATION|Coverage|Properties'
#   ^\s*index\s*<\s*l->size\s*$
```

Classification of the 3503 lines:

| Family | Count | Migration sensitivity |
|---|---:|---|
| Pure verdict (`VERIFICATION SUCCESSFUL/FAILED`) | 3344 | none |
| + standard checker markers (`Violated property`, `dereference failure`, `data race`, …) | ~20 more | none — checker message strings, symbol-independent |
| Coverage / property **counts** (`Properties: N verified`, `Branch Coverage: 100%`, `Reached : N`, `Total Asserts: N`) | 39 | safe **iff** the migration does not change the number of properties/branches/asserts (Phase 4.1/4.4 must not) |
| Frontend **diagnostics** (`TypeError: foo() missing … 'x'`, `ERROR: Variable x … uninitialized`, `Unsupported …`) | 46 | embed **Python source identifiers**, which the migration preserves (it changes IR construction, not the frontend's source-named diagnostic text) |
| Source-derived assertion/property text (`x is equal to y` = the `assert` message; `step != 0` = source var) | 2 | safe — Python source literal / source name, not IR pretty-print |
| **Pretty-printed IR expression** (`index < l->size`) | **1** | **the one genuine RP10 gate** — IR-pretty-printer output for the list-bounds guard under `--ir`; depends on member-access construction + symbol naming staying byte-identical |
| Concrete counterexample **variable values** (`State N`, `var = value`) | **0** | not a gate — no Python test pins trace values |

**Conclusion.** Symbol-naming / pretty-print stability is a **real but
narrow** Phase 4.4 gate. There is exactly **one** canonical guard test —
`regression/python/bounds-checking_fail` (`index < l->size`, `--ir`, the
native list-bounds path) — that pins pretty-printed IR expression text; keep
its matched line byte-identical when `python_list.cpp`'s index/bounds
construction moves to `index2t`/`member2t`. Trace-value stability is **not**
required (0 tests). The coverage-count (39) and frontend-diagnostic (46)
families are covered by the §10 "matched-text identical" gate as long as it
captures matched lines (not just the verdict) and the property/branch/assert
counts are held invariant — both already mandated by §10. This makes RP10
cheap to discharge: one targeted bounds test plus the existing matched-text
gate, rather than a diffuse symbol-naming audit.

### 15.7 Phase 4.3 byte-identity audit — which type families ride the seam losslessly

The Phase 4.3 invariant is byte-identical legacy output at the seam (the
elementary/array commits and their `irep2_type_roundtrip_test` cases assert
exact `==`). A spike over `lower_to_seam(t) = migrate_type_back(t)` for every
`type_handler` builder shape established **which families can hold that bar**,
because `lower_to_seam` is byte-identical **only** when the legacy form
already carries every field `migrate_type_back` re-emits:

| Builder shape | Byte-identical? | Why |
|---|:---:|---|
| Elementary (`signedbv`/`unsignedbv`/`bool`/float) | ✅ | scalar, no extra fields |
| `array_typet` | ✅ | subtype + size only |
| `pointer_typet(value)` / `pointer_typet(symbol)` | ✅ | subtype only; `carry_provenance` default `false` matches a fresh `pointer_typet` |
| NoneType/Optional (`pointer_type()` = pointer-width `unsignedbv`) | ✅ | elementary in disguise |
| `complex` struct (3-arg components + `tag`) | ✅ | `tag`/`pretty_name` already present |
| **Tuple** struct (2-arg components, **tagless**) | ❌ | `migrate_type_back` **adds** an empty `tag` **and** an empty `pretty_name` per component |
| **Optional** struct (`set_access("public")`) | ❌ | `migrate_type` carries only types/names/pretty-names/tag/packed — component **`#access` is dropped** |
| **Callable** (`pointer_typet(code_typet)`, **no args**) | ❌ raw, ✅ IREP2 | `migrate_type_back` **adds** an empty `arguments` sub the source lacks; `migrate_type` canonicalises it away, so the IREP2 type is identical (migrated — see below) |

**Consequences for the family order (§7 Phase 4.3).**

- The **pointer family** was migrated **ahead of struct/class** (commit
  `[python] Phase 4.3: migrate pointer-family type builders to IREP2`):
  NoneType/Optional → `unsignedbv_type2tc`, the list/annotation pointees →
  `pointer_type2tc`, the generic list type → `pointer_type2tc` +
  `symbol_type2tc`. The doc's family sequence is an ordering preference, not a
  hard dependency (§8: 4.3 only blocks 4.4); reordering to "clean families
  first" keeps every commit byte-identical.
- The **tuple/optional struct builders are F-P5 seam-attribute cases** — the
  same class as `#cpp_type`/`#member_name`. They cannot ride the IREP2 type
  losslessly and belong to **Phase 4.5** (the explicit hand-off that
  re-attaches seam attributes), not a Phase 4.3 internal migration. The only
  lossless `type_handler` struct is `complex` (but it is a cached header-inline
  free function feeding expression construction, so it tracks with 4.4/4.5).
- The **Callable** builder **was migrated** (commit `[python] Phase 4.3:
  migrate Callable function-pointer type to IREP2`): a no-argument
  `code_type2tc` (empty arg/name vectors satisfy the `args.size() ==
  argument_names.size()` invariant) wrapped in `pointer_type2tc`, lowered at
  the seam. It is the one pointer/function-family case that is **not raw
  byte-identical** — the legacy source never touched `code_typet::arguments()`,
  so `migrate_type_back` adds an empty `arguments` sub — but it **is
  IREP2-equivalent** (`migrate_type` canonicalises the empty `arguments`, so
  `migrate_type(migrated) == migrate_type(legacy)`, the harness's documented
  level-1 contract) and **GOTO-identical** (goto-convert normalises the
  leftover sub; verified end-to-end on both solvers, all 10
  `regression/python/callable*` tests pass). This is the precedent that raw
  byte-equality is a *bonus* over the stated IREP2-equivalence bar, and that a
  field `migrate_type_back` adds can still be behaviour-inert once it reaches
  the GOTO program. Real-parameter function *symbol* types (≥1 argument, built
  in the converter, not `type_handler`) already carry the `arguments` sub and
  are raw byte-identical; they migrate with the Phase 4.4/4.5 symbol work.

Reproduce: extend `irep2_type_roundtrip_test` with
`migrate_type_back(migrate_type(t)) == t` (raw) and `migrate_type(migrated) ==
migrate_type(legacy)` (IREP2) over each shape above; the raw ❌ rows fail the
first but the Callable row passes the second.

## 16. Converter-stage construction constraint — the pre-`adjust` resolved-type wall (F-P10 / RP14)

> Discovered while executing Phase 4.4 on `converter_expr.cpp` (the hub, file 2
> of §7.2.2), 2026-06-02. This is a **load-bearing correction to the §7.2.2
> premise**: not every expression-construction idiom is migratable at the
> converter stage. Recorded in the §15.1 spirit — a wrong difficulty/ordering
> estimate in a verification plan is the defect the plan exists to prevent.

**F-P10 — `member2t` and `index2t` carry resolved-type preconditions the
converter cannot satisfy.** `member2t`'s constructor asserts
`source->type ∈ {struct, union, complex}` (`irep2_expr.h:1502`); `index2t`
asserts `is_array_type(source) || is_vector_type(source)` (`irep2_expr.h:1576`).
The legacy `member_exprt` / `index_exprt` constructors assert **nothing** — they
tolerate a `source` whose type is still a **`symbol`** (an unresolved struct/array
reference). The Python converter builds expressions **before** `clang_cpp_adjust`
runs (`python_language.cpp:245`, the §3 seam) and therefore routinely constructs
member/subscript access over a `symbol`-typed base that only `ns.follow()`s to a
struct/array **later**. So a mechanical `member_exprt → member2tc` /
`index_exprt → index2tc` substitution **aborts at conversion time** on inputs the
legacy path handled.

**Evidence (reproduced, then reverted).** Migrating the two attribute-access
`member_exprt` sites in `converter_expr.cpp` (≈510, ≈1002) to `member2tc` —
following the §7.2.1 idiom table verbatim — fired, under the asserts build:

```
Assertion failed: (source->type->type_id == type2t::struct_id || ... union ...
complex), function member2t, file irep2_expr.h, line 1502.
```

on `regression/python/type-annotation-class` (`class Point: __init__ → self.x=i;
points=[Point(0)]`) and ~10 sibling `type-annotation-class*` /
`typing_tuple_attr_annotation*` tests. The asserts-enabled build caught it at
construction — exactly the IREP2 soundness benefit (a mis-typed node fails to
build, not at symex), and exactly why every Phase 4.4 commit **must** run on an
asserts build (§10.3).

**RP14 (new risk) — mechanical idiom substitution over symbol-typed bases is a
hard abort, not a silent miscompile.** Severity: **high** for effort/ordering
(it invalidates the "files 1–8 are mechanical" framing for the *attribute and
subscript* portions of the hub), **low** for soundness (it cannot ship — it
crashes the asserts build and any asserts-enabled CI).

**The refined migration rule (supersedes the §7.2.1 "apply at each call site"
framing for member/index):**

> Migrate only subtrees **rooted in already-resolved, concrete types** —
> operational-model call results (`__ESBMC_list_size`, `strlen`), sizes,
> primitives, pointers — using the **assert-free** IREP2 kinds (`symbol2t`,
> `typecast2t`, `address_of2t`, `dereference2t`, `equality2t`,
> `side_effect_function_call2t`). Do **not** migrate `member2t`/`index2t`
> construction whose source is a Python class/list/dict value, because that
> source is `symbol`-typed until `clang_cpp_adjust` (P2) follows it. This is
> precisely why the `converter_unop.cpp` truthiness subtrees migrated cleanly
> (#5041, #5045): their operands are `size_type`/`bool`/`strlen`-typed concretes,
> not symbol-typed Python objects.

**Consequence for the §7.2.2 order.** The attribute/subscript construction in
`converter_expr.cpp` (member/index sites) and the string-char index/deref in
`converter_compare.cpp` are **blocked at the converter stage** and move only
once one of these enabling steps lands:

| Option | What | Cost / status |
|---|---|---|
| (a) Type-following at construction | Build `member2tc`/`index2tc` over a source whose type is `migrate_type(ns.follow(base.type()))` | Unsound as a naive retype — the *expression's* type is intrinsic; you cannot restamp a `symbol2t`'s type without lying about the node. Needs a real "resolve then build" helper, and must prove the followed type equals what `adjust`+goto-convert would have produced. **Open design question — investigate before attempting.** |
| (b) Defer to post-`adjust` | Migrate member/index where types are already resolved (a different pass than the converter) | Touches P2 (the shared `clang_cpp_adjust` boundary) — **out of scope** for Part IV (§Out-of-scope pins). |
| (c) Restrict Phase 4.4 to assert-free / resolved-type subtrees | Cherry-pick the migratable subtrees per file; leave member/index legacy at the seam | **Recommended.** No new infra, behaviour-preserving, matches the merged #5041/#5045 shape. The hub's *non*-attribute construction (e.g. the numpy `.shape` length subtree, an assert-free `typecast(size_call,int)`) qualifies. |

**The `.shape` caveat (test-coverage gap, not a soundness one).** The one
assert-free migratable subtree found in `converter_expr.cpp` — the numpy list
`.shape` length `(int)__ESBMC_list_size(&base)` (≈481–490, ≈830–837) — was
migrated successfully (built, verified SUCCESSFUL) but **could not be gated**:
the path is numpy-only (plain-list `.shape` is rejected at type-check), there are
**zero** `import numpy` tests in `regression/python`, and numpy is not installable
under the CPython-sanity harness (`scripts/check_python_tests.sh`). Shipping
un-gateable code violates the §1 constraint, so it was reverted. **Gap to
resolve before migrating it:** decide a numpy regression-test strategy (is numpy
in CI's CPython? should numpy tests skip CPython sanity?) — investigate, do not
guess.

**Re-verification commands:**
```sh
grep -n 'type_id == type2t::struct_id' src/irep2/irep2_expr.h   # member2t assert (~1502)
grep -n 'is_array_type(source)\|is_vector_type(source)' src/irep2/irep2_expr.h  # index2t assert (~1576)
grep -rlE 'import numpy|np\.array' regression/python/*/main.py | wc -l           # 0 — no numpy coverage
```

### 16.1 The deeper wall — `migrate_expr` of a converter-stage operand recurses into the assert (F-P11)

> Discovered executing Phase 4.4 on `converter_binop.cpp` (file 4 arithmetic),
> 2026-06-02, via a **second independent route** to the §16 assert — which is
> what promotes this from a per-file quirk to a **general feasibility limit on
> Phase 4.4**.

**F-P11 — you cannot `migrate_expr` an arbitrary converter-stage operand.** The
§16 finding is broader than "don't build `member2t`/`index2t`." The standard
Phase 4.4 recipe (§7.2.3) is *migrate the legacy operands forward, compose the
typed node, back-migrate once*. But `migrate_expr` is **recursive**: migrating
an operand that **is or contains** a `member`/`index` access over a still-`symbol`
type builds `member2t`/`index2t` **inside the migration** and trips the same
`irep2_expr.h:1502`/`:1576` assert. At the converter stage (pre-`adjust`) those
unresolved member/index sub-expressions are pervasive — `self.x`, `obj.attr`,
`a[i]` are ordinary operands of arithmetic, comparison, and assignment.

**Evidence (reproduced, then reverted).** Migrating **integer `Add`/`Sub`/`Mult`**
in `build_binary_expression` (`converter_binop.cpp:1383`) — assert-free
`add2tc`/`sub2tc`/`mul2tc` over operands first coerced to a common bitvector
type, the *safest possible* arithmetic migration — aborted **277 of 432**
arithmetic/numpy tests. The abort was **not** in the arith node (which is
assert-free) but in `migrate_expr(lhs)` / `migrate_expr(rhs)` when an operand
carried an unresolved member access (e.g. `regression/python/math-sqrt-any-typed`:
a `self`-attribute operand), firing the identical `member2t … line 1502`. A
trivial `7 + 3` smoke test passed — masking the problem until the corpus ran.

**Why the file-1 subtrees were safe and general construction is not.** The
`converter_unop.cpp` truthiness subtrees (#5041/#5045) migrated cleanly because
**every operand was a fully synthetic, concrete-typed value** — `strlen(...)` /
`__ESBMC_list_size(...)` results, `size_type` symbols, freshly-built constants —
never a member/index access over a Python object. `migrate_expr` of those never
reaches a `member2t`/`index2t`. General-expression construction has no such
guarantee.

**Consequence — the migratable surface at the converter stage is small and
specific.** A converter-stage construction is safely migratable **only when
every operand it migrates forward is synthetic and concrete-typed** (OM-call
results, sizes, constants). Any construction whose operands can be arbitrary
user sub-expressions — i.e. the bulk of Phase 4.4 (the 842 `symbol_expr`, 121
`member`, the arithmetic/comparison builders) — **cannot** use the
migrate-forward/back-migrate recipe until types are resolved. Type resolution
happens in `clang_cpp_adjust` (P2), which Part IV holds **out of scope**.

**Go/stop recommendation (this is the §4.6/§12 decision point).** Phase 4.4's
"build expression construction in IREP2 internally" goal is **feasible only for
synthetic-operand subtrees**, which file 1 has now exhausted in the unary path.
The honest end-state mirrors Part III's Solidity outcome (§14): **the frontend
stays legacy at the construction stage by design**, because IREP2's
resolved-type invariants cannot be met before `adjust`. Two forward paths, both
larger than Part IV's frontend-only scope:

| Path | What | Scope |
|---|---|---|
| (i) Resolve-then-build | Run type resolution (follow symbol types) **before** construction, so operands migrate cleanly | Moves/duplicates `adjust`-stage work into the converter — a real architectural change, **P2-class**, out of Part IV scope. Must prove the resolved type equals what `adjust`+goto-convert produce. |
| (ii) Stop at file 1 | Declare Phase 4.4 complete-as-feasible: synthetic-operand subtrees migrated (`converter_unop`), general construction stays legacy at the seam | **Recommended.** Behaviour-preserving, matches the merged shape, and is the same "encapsulate, stay legacy at the seam" conclusion Part III reached for Solidity. |

**Re-verify:**
```sh
# integer-arith migration in build_binary_expression aborts the corpus, not just one test:
#   reproduce by mapping Add/Sub/Mult -> add2tc/sub2tc/mul2tc over migrate_expr(lhs/rhs),
#   then: ctest -R 'regression/python/.*(arith|np_|math)' --timeout 60   # ~277/432 abort
grep -n 'migrate_expr' src/util/migrate.cpp | head        # recursive; no resolved-type guard
```

## 18. Pause record — why Part IV stops, and the round-trip it cannot remove

> **Status: PAUSED 2026-06-09 by an explicit go/stop decision.** Recorded
> in the §15.1 / §16.1 honesty spirit: a verification plan that keeps
> producing commits with **zero net pipeline effect** is churn, and saying
> so plainly is part of the contract.

### The observation that triggered the pause

Every Phase 4.4/V.3 converter helper has the shape:

```cpp
exprt build_X(const exprt &legacy_in /*…*/)
{
  expr2tc in2;  migrate_expr(legacy_in, in2);          // irept  -> expr2tc
  return migrate_expr_back(X2tc(/*…*/, in2));          // expr2tc -> irept
}
```

Legacy in, legacy out. The `expr2tc` exists only inside the call. The IR
that reaches `clang_cpp_adjust`, `goto_convert`, and symex is
**byte-identical legacy `irept`** — which is exactly why every commit was
verdict- and matched-text-preserving. **Net IREP2 in the verified
pipeline: zero.** Even the plan's "correct" form (build a whole subtree,
back-migrate *once* at its root, §7.2.3) is still a round-trip; it only
minimizes the *count* of `migrate_*` calls, not the round-trip itself.

### Why it is structurally forced (the consumers are legacy)

The converter's output is consumed by stages that are still legacy, so
anything built in IREP2 must be lowered back before it crosses the seam:

- **Function bodies are legacy `codet` (W1/P1).** `convert_function`
  reads `to_code(symbol.get_value())` (`goto_convert_functions.cpp:129`)
  and `goto_convertt::convert` dispatches on `code.get_statement()`
  strings (`goto_convert.cpp:220`), migrating each expression operand to
  `expr2tc` **per-instruction** as it builds the goto program. IREP2 has
  no structured-CF code kinds, so the frontend cannot hand it an IREP2
  body.
- **The symbol table receives legacy.** `create_symbol` →
  `set_type(typet)`; `grep -rn 'set_value(.*2tc' src/python-frontend` → 0.

So the back-migration is not removable by any amount of frontend work.

### What did land (Phase 4.4/V.3), and its honest value

The converter now builds a large share of its expression construction in
IREP2 internally, behind per-file `build_*` helper suites
(`function_call/expr.cpp`, `builtins.cpp`, `complex_handler.cpp`,
`python_math.cpp`, `numpy_call_expr.cpp`, the OM handlers, etc.), each
lowering with one `migrate_expr_back` at the subtree root. The merged
PRs span #5041/#5045 (file-1 truthiness) through the V.3 series
(#5235–#5263). **Value delivered:** (a) compile-time-typed construction
inside the converter — a mis-built node fails to compile, not at symex;
(b) the legacy-builder call sites are pre-positioned so that, once V.4
makes bodies IREP2 and V.6 flips the symbol writes, the back-migrate at
each subtree root is the *only* thing left to delete. **Value not
delivered:** any change to the verified IR, or any reduction in
round-trips. That awaits V.4/V.6.

### Census at the pause

`src/python-frontend`: **4907** legacy IR mentions, **245** IREP2
(`*2tc`) — up from 31 at the prior reconcile and ~6 pre-4.3. The IREP2
surface grew *inside* the converter while the seam back-migrates, so
external bytes are unchanged. `set_type(2tc)`/`set_value(2tc)` at the
symbol-write boundary: still **0** (W1/F-P1 intact).

### The decision

Part IV is **paused, not abandoned**. The frontend-scoped,
behaviour-preserving target it set ("IREP2-internal where the
resolved-type/CF invariants allow, legacy at the seam") is essentially
reached for the clean construction surface. Continuing to migrate the
residual one-off sites would add round-trips for no pipeline benefit.
The real next step is **Part V Phase V.4** — give IREP2 the structured
control-flow code kinds and teach `goto_convert` to consume IREP2 bodies,
so the body-seam back-migration disappears. Active work moves there.

# Part V — Python frontend → **100 % IREP2** (the deep-pin program)

> **Status: in progress.** Phase V.4 (IREP2-native bodies through
> `goto_convert`) has landed V.4.0→V.4.3 across all frontends; **V.4.4a
> flipped the flag on by default** (full-corpus verdict-parity sweep clean —
> Bitwuzla + Z3; Python, C, C++ at every standard, floats, cstd, CUDA,
> Solidity) and **V.4.4b removed the legacy body seam entirely** — the IREP2
> round-trip is now the only body path, the `to_code(symbol.get_value())`
> bypass and the `--no-irep2-bodies` escape hatch are gone, and
> `--irep2-bodies` survives only as a deprecated no-op. The deeper W1 removal
> (rewrite `goto_convert_rec` handlers to consume `code_*2t` directly, dropping
> the round-trip) was scoped and prototyped and is **concluded as a closed
> boundary** — the body round-trip is fundamental for source-location fidelity
> (see *V.4 outcome — the deeper W1 removal is a closed boundary*). Remaining:
> the V.1k IREP2-native adjuster keystone and V.5/V.6. This supersedes the
> "not started / costed path"
> framing this paragraph originally carried (it predated the V.4 work); the
> §V.2 phase entries hold the authoritative per-phase status, and §18's
> "next step is Part V Phase V.4" pause note is likewise discharged.
> Part IV (§§1–16.1) is the *frontend-scoped, behaviour-preserving* plan and
> its honest conclusion is "encapsulate and
> IREP2-internalise what the resolved-type / control-flow invariants allow;
> stay legacy at the seam by design" (§16.1 go/stop). Part V answers a
> *different* question the user posed — **"what would it take to make the
> Python frontend 100 % IREP2?"** — and the answer is a **repo-wide
> pipeline program**, not a frontend sweep. This part exists so that the cost
> and shape of that program are on record; **read §V.1 before treating any
> of it as actionable.** Every anchor is a snapshot at the revision this lands
> on (Part IV's drift caveat applies); re-run the cited command before acting.

## V.1 What "100 %" means, and why it is not a frontend task

**Definition (the acceptance bar for the whole program):**
1. `git grep -P '\b([A-Za-z_]*(exprt|typet|codet)|irept)\b' -- src/python-frontend`
   → **~0** (modulo unavoidable boundary glue, explicitly enumerated at close).
2. The converter writes the symbol table **only** via `set_type(type2tc)` /
   `set_value(expr2tc)` (`symbol.h:49-67`, the B2 IREP2 setters) carrying
   **fully-IREP2** values — `grep -rn 'set_type(\|set_value(' src/python-frontend
   | grep -vc 2tc` → **0**.
3. Function **bodies** are IREP2; `goto_convert` consumes them without a
   `migrate_*` back-hop (today: legacy at `goto_convert_functions.cpp:116`,
   `to_code(symbol.get_value())`).
4. No `#`-attribute escape hatch on a legacy `irept` survives into a shared
   downstream pass (F-P5).

**Why this is structurally a pipeline program, not frontend work.** The Python
converter's output is consumed by **shared, still-legacy** passes. You cannot
hand them IREP2 until they are IREP2-native, and they are not Python's to own.
The four walls (all *proven and merged* in Part IV §16/§16.1, and re-anchored
here) sit **outside** `src/python-frontend`:

| Wall | Statement | Anchor (re-verify) | Owner |
|---|---|---|---|
| **W1 (P1)** | IREP2 has the **flat goto-level** code kinds (`code_block`/`assign`/`decl`/`return`/`goto`/`skip`/`dead`/`free`/`expression`) but **no structured CF kinds** (`ifthenelse`/`while`/`for`/`switch`/`break`/`continue`/`label`); `goto_convert` reads the body as **legacy** `codet`. | `grep 'IREP2_EXPR(code' src/irep2/expr_kinds.inc`; `goto_convert_functions.cpp:116` | shared (all frontends) |
| **W2 (P2 / F-P10 / F-P11)** | `member2t`/`index2t` assert a **resolved** struct/array source (`irep2_expr.h:1502`/`:1576`); the converter runs **before** `clang_cpp_adjust` follows `symbol`-typed bases; `migrate_expr` recurses into the assert (§16.1). | §16, §16.1 | shared (`clang_cpp_adjust`) |
| **W3 (F-P5)** | `#cpp_type`/`#member_name`/`#cformat` are read off **legacy** nodes by `clang_cpp_adjust_expr.cpp:464`, `cpp_expr2string.cpp:138-140`, `goto2c/expr2c.cpp:169-174`. | §15.2 | shared |
| **W4** | The counterexample C/C++ printer consumes `#cpp_type`/`#cformat` (Part II 2.7, "out of scope"). | Part II §2.7 | shared |

> **The reframing (load-bearing):** *"Make the Python frontend 100 % IREP2"*
> ≡ *"Make ESBMC's frontend→goto pipeline IREP2-native, then flip Python
> last."* This is **finishing** the repo-wide IREP2 migration (#4715) through
> `goto_convert` and `clang_cpp_adjust` — the boundaries B1/P1/P2 that Parts I
> and III closed **on purpose** because they are shared and high-blast-radius.
> Part V is therefore an **umbrella-issue proposal**, and the Python flip
> (Phase V.3) is its smallest, last step.

## V.2 The phased program

Ordered by dependency. Phases V.1k (keystone), V.2, V.4, V.5 are **shared
infrastructure** — they change C, C++, CUDA, Solidity, and Java/Kotlin lowering,
not just Python. Each phase holds Part IV's universal gate **plus `esbmc-cpp`**
(the C++ suite) wherever it touches the shared clang-cpp/goto passes.

### Phase V.0 — Goal, harness, baseline
Pin the §V.1 acceptance bar. Stand up the dual-solver (Bitwuzla + Z3),
asserts-build, verdict + matched-text gate over the full `regression/python`
strata **and** the operational-model `.py` corpus (§3.1), plus `esbmc-cpp` as a
shared-pass backstop. Snapshot the census.
*Accept:* harness reproduces today's verdicts/text; census tooling agreed; the
4 walls each have a live reproducer (the §16/§16.1 asserts).

### Phase V.1k — Resolve-then-build (removes W2) — **the keystone**
*Nothing in the converter's expression construction can migrate before this
(§16.1).* Introduce IREP2-native type resolution so that, at construction time,
`member`/`index`/general operands carry **resolved** struct/array types and
`migrate_expr`/`member2tc`/`index2tc` no longer abort. Two designs to prototype
and compare head-to-head before committing:
- **(a) pre-construction resolver** in the frontend (follow `symbol` types
  before the converter builds nodes), or
- **(b) IREP2-native `clang_cpp_adjust`** (a Python-specific adjuster the
  converter feeds, replacing the legacy round-trip).

**Proof obligation (non-negotiable):** the resolved type must equal what
today's `clang_cpp_adjust` + `goto_convert` produce — prove by round-trip
equivalence over the corpus on an asserts build. *Accept:* a representative
member/subscript expression builds `member2tc`/`index2tc` **in the converter**
with **zero** `irep2_expr.h:1502/1576` aborts across the whole suite; verdict +
text unchanged.
**Investigation gap (resolve first):** map exactly where/how `clang_cpp_adjust`
resolves Python attribute/element types today (`grep -rn 'follow\|#cpp_type'
src/clang-cpp-frontend/clang_cpp_adjust*`); decide whether that resolution can
run pre-construction soundly, or whether construction must move below adjust
(which makes the converter an adjust client — design implication for (a) vs (b)).

#### V.1k spike outcome (round 1) — feasible; mechanism pinned; lean to (b)

A read-only spike (2026-06-02) settled the feasibility question and the
resolution locus. **Verdict: design (a) is feasible, but (b) is the cleaner
target.** Findings, each with its anchor:

1. **The mismatch is exact.** `migrate_type(symbol_typet)` returns
   `symbol_type2tc` **by-name — it does not follow** (`migrate.cpp:188-189`).
   `member2t` asserts its source is a **followed** `struct`/`union`/`complex`
   *and* that the named component exists (`irep2_expr.h:1499-1505`); `index2t`
   asserts `is_array_type`/`is_vector_type` (`:1576`). A base arriving as
   `symbol_type2t` therefore aborts at construction — the §16/§16.1 wall.
2. **Bases are symbol-typed by design.** Python class instances are stored as
   `pointer→symbol_typet("tag-Cls")`, explicitly "resolved lazily via
   `ns.follow()` at use time" (`converter_class.cpp:273-279`).
3. **Resolution happens in `clang_cpp_adjust`, not in `migrate` or the
   converter.** `goto_convert`'s `migrate_expr` builds
   `member2tc(type, migrate(base), comp)` (`migrate.cpp:1534-1541`) with **no
   follow**; `clang_c_adjust::adjust_member` only auto-derefs pointers/arrays
   (`clang_c_adjust_expr.cpp`, the `adjust_member` body). The symbol→struct
   completion is the `ns.follow`-based resolution inside `clang_c_adjust`
   (`clang_c_adjust_expr.cpp:154,324,446,475,541,…`), which runs **after** the
   converter (`python_language.cpp:245`) and **before** goto-convert. That is
   the P2 boundary, precisely.
4. **Design (a) is feasible** — the converter already holds the namespace and
   the structs are registered (resolvable at use time by design, finding 2), so
   it *can* `ns.follow` a base to its struct and build `member2tc`/`index2tc`
   with the resolved `type2tc`. **RV2 proof obligation, made concrete:** prove
   the converter-resolved type equals the `clang_cpp_adjust`-resolved type by
   GOTO-byte + dual-solver + asserts equivalence over the corpus.
5. **The (a)-vs-(b) decision, sharpened — lean (b).** `clang_cpp_adjust` does
   type-following **and** pointer auto-deref **and** the `#cpp_type` reads (W3)
   in one legacy pass. Replicating only the following converter-side (a) risks
   duplicating resolution logic and leaving adjust running for the rest — a
   two-places-resolve hazard. An **IREP2-native adjust (b)** that does
   follow + deref + attribute-carriage in one IREP2 pass collapses **V.1k + V.2
   + W3** into a single deliverable and avoids the duplication. Provisional
   recommendation: pursue **(b)**, scoped as an IREP2-native Python adjuster.

**Open sub-question for spike round 2 (the prototype):** pin the *exact* line at
which a member/subscript base's `symbol_typet` is replaced by the resolved
`struct_typet` on the node reaching goto-convert (confirm empirically with
`esbmc <t>.py --goto-functions-only` on `type-annotation-class`, inspecting the
member base type), then build the smallest converter-side resolve-then-build
experiment that discharges or refutes RV2 on that one case.

#### V.1k spike outcome (round 2) — design (a) **REFUTED**; (b) confirmed required

A working prototype of design (a) settled RV2 empirically (2026-06-02, asserts
build, branch `spike/v1k-resolve-then-build-prototype`, **not merged** — it
regresses). The prototype made the **one** change (a) requires at the
attribute-access site (`converter_expr.cpp` ≈494-510): follow the dereferenced
pointee so the base carries the resolved struct, then build `member2tc` over it
and lower once:

```cpp
deref.type() = ns.follow(member_base.type().subtype());   // was: .subtype() (symbol_typet)
// member_base.type() is now the resolved struct
expr2tc src2; migrate_expr(member_base, src2);            // src2->type is struct
return migrate_expr_back(member2tc(migrate_type(clean_type), src2, attr_name));
```

**Result:** the flat-member cases that aborted in §16/§16.1 now pass
(`type-annotation-class`, `math-sqrt-any-typed`, a `p.x==5` probe → all
SUCCESSFUL, no assert). **But the corpus refuted it: 20 of 488** attribute/class
/arith tests **regressed**, *entirely* in four families:

| Regressing family | # | What a single leaf-follow misses |
|---|---:|---|
| `nested-attr-*` (incl. chains) | 8 | `self.b.a`: the **inner** member is built over an **outer** member whose type a leaf-follow did not resolve — adjust follows at **every** level, recursively. |
| `github_4117_*` (attr type-inference) | 8 | attribute **shadowing / conflict / `None`-typed** inference — adjust resolves these via more than a struct-name follow. |
| `dataclass_*` | 2 | dataclass field access has adjust-stage handling beyond a plain `ns.follow`. |
| `self_ref_nested_attr_chain*` | 2 | self-referential chains — same recursive-resolution gap as nested-attr. |

**Verdict — RV2 is *not* dischargeable by design (a).** A converter-side
single `ns.follow` replicates only the **flat, single-member** slice of
`clang_cpp_adjust`'s resolution; nested chains, dataclass fields, and the
attribute type-inference edge cases need adjust's **full recursive,
dataclass-aware, inference-aware** completion. Replicating *all* of that in the
converter would duplicate `clang_cpp_adjust` piecemeal and brittlely — exactly
the two-places-resolve hazard flagged in round 1.

**Consequence (decision recorded):** **adopt design (b)** — an **IREP2-native
adjuster** that runs adjust's *complete* resolution (recursive following +
auto-deref + dataclass/inference handling + `#cpp_type`/`#member_name` carriage)
as one IREP2 pass, replacing the legacy `clang_cpp_adjust` round-trip on Python
output. This is the keystone deliverable for V.1k and **also retires W3**
(§V.2): the same pass that resolves types is the natural home for IREP2-native
attribute carriage. The 20-test regression set above is the **acceptance
fixture** for the (b) adjuster — it must take all four families back to green
while keeping the flat-member improvement.

Reproduce: re-apply the prototype diff above and run
`ctest -R 'regression/python/.*(attr|class|nested|dataclass|github_4117|math)'`
— 20 failures, all in the four families; revert restores green.

#### V.1k design (b) — IREP2-native Python adjuster: scope against the 20-test fixture

The round-2 decision is **design (b)**: replace the legacy `clang_cpp_adjust`
round-trip on Python output with an IREP2-native adjuster that runs the
*complete* resolution as one IREP2 pass. This section specifies it against the
20-test acceptance fixture.

**What the fixture requires (resolution operations, by family):**

| Family (count) | Representative | Resolution the adjuster must perform |
|---|---|---|
| `nested-attr-*` (8) | `self.b.a.get_value()` | **Recursive** struct-following: building `X.a` needs `X` (itself member `self.b`) already resolved to a struct. A single leaf follow (design (a)) handled only the outermost level — the exact gap that sank (a). |
| `github_4117_*` (8) | `n1.next = n2; n1.next.value` (`next: None`→`Node`) | Honor the Python frontend's **attribute type inference** (`python_annotation` + flow-sensitive `flow_class_map_`): follow the *inferred* struct type, not the declared `None`. |
| `dataclass_*` (2) | `fields(C)`, `is_dataclass(c)` | **Dataclass struct/field resolution** (fields are struct components via the `dataclasses` model). |
| `self_ref_nested_attr_chain*` (2) | self-referential chains | Same recursive-following gap as `nested-attr`. |

Net: recursive `symbol_type2t`→struct following + auto-deref (already in
`adjust_member`) + consume the converter's inferred types + dataclass handling —
i.e. `clang_cpp_adjust`'s complete completion, IREP2-native.

**The core design problem — chicken-and-egg, and its resolution.**
`member2t`/`index2t` assert a resolved struct/array source **at construction**
(`irep2_expr.h:1499-1505`/`:1576`, `#ifndef NDEBUG`). So the converter cannot
build the IREP2 node before resolution, yet the adjuster needs nodes to resolve.
Resolution is a **two-phase source invariant**:
1. **Relax** the construction assert to permit a `symbol_type2t` source *pending
   resolution* (localized IREP2-core change at those two sites). The ~11
   symex/simplifier consumers (`to_member2t(...).source_value`,
   `struct_union_members(p->type)`, `to_struct_type(type).members[i]` — e.g.
   `goto_symex_state.cpp:281`, `symex_main.cpp:1309`, `memory_ops.cpp:261`) need
   **no change**: they all run strictly **post-adjust**.
2. **Post-adjust verification:** the adjuster asserts, at exit, that every
   `member2t`/`index2t` source is now a resolved `struct`/`union`/`array` —
   restoring the strong invariant before symex/goto see any node.

**Deliverable shape.**
- *Where:* replaces the legacy `clang_cpp_adjust` round-trip on Python output
  (`python_language.cpp:245`); walks the whole symbol table, mirroring
  `clang_c_adjust::adjust()`.
- *What, per IREP2 node:* recursively resolve `member2t`/`index2t` sources
  (`symbol_type2t` → followed `struct_type2t`/`array_type2t` via the namespace),
  auto-deref pointer sources, and carry `#cpp_type`/`#member_name` IREP2-natively
  — **folding in W3 (§V.2)**, since the resolving pass is the natural home for
  attribute carriage.
- *Reuses:* the converter's already-computed inferred types — it does not
  re-infer; it follows what `python_annotation`/`flow_class_map_` produced (which
  is why `github_4117` resolves once following is recursive).

**Acceptance criteria.**
1. The **20-test fixture green** *and* the flat-member improvement retained
   (`type-annotation-class`, `math-sqrt-any-typed`).
2. Full `regression/python` + model `.py` corpus: verdict + matched-text parity,
   **both solvers**, asserts build.
3. **`esbmc-cpp` green** (the relaxed assert touches C++ too — RV3).
4. **GOTO-byte parity** vs the legacy adjust path.

**Open investigations before coding (resolve first):**
- **A.** Confirm the relaxed assert has no *other* consumer that builds
  `member2t` and reads its struct source *before* adjust (the shared clang-cpp
  frontend builds `member2t` too).
- **B.** Pin where the converter's inferred attribute type lands on the node, so
  the adjuster follows the right symbol id for the `github_4117` flow cases
  (trace `flow_class_map_` → the member source type).
- **C.** Decide **Python-only pilot** adjuster first (lower blast radius) vs the
  shared IREP2-native adjust from the start (RV3). Recommend the Python-only
  pilot, then generalize.

#### V.1k investigation outcomes (A/B/C) — the (b) core change is low-risk

Read-only investigations (2026-06-02) run to harden the (b) spec before coding.

**A — relaxing the construction assert has *zero* blast radius on existing
frontends. ✅** There are **no** `member2tc(` construction sites in any frontend
(`grep -rn 'member2tc(' src/{clang-c,clang-cpp,python}-frontend` → empty): the
frontends build **legacy** `member_exprt`, and `member2t` is constructed only at
goto-convert (via `migrate.cpp:1534-1541`) and inside symex — **both
post-adjust**. The C++ pipeline is `convert → adjust → goto`
(`clang_cpp_language.cpp:123,127`), identical to Python. Therefore `member2t` is
built pre-adjust **only** in the new Python converter path; for every existing
path it is already post-adjust over a resolved source, so the relaxed assert
changes nothing for them. The two-phase invariant's risk surface is exactly the
new Python window the adjuster closes — *not* the rest of the tree. This
substantially de-risks the one IREP2-core change (b) needs.

**B — the converter's inferred attribute type is already on the node as the
`symbol_typet` tag; the adjuster needs no inference logic. ✅** The Python
attribute inference (`flow_class_map_`, `python_converter.h:1078`, consumed at
`converter_expr.cpp:590`; `infer_attr_type_from_usage`,
`converter_class.cpp:659`) writes the inferred class onto the node as
`gen_pointer_type(symbol_typet("tag-<inferred-class>"))`
(`converter_class.cpp:279`). So for the `github_4117` flow cases
(`n1.next = n2; n1.next.value`, `next: None`→`Node`), the member source already
carries `symbol_typet("tag-Node")` — the adjuster simply follows that tag to the
resolved struct. Design (a) missed these only because it patched **one** site
(≈510) while these flow through the `flow_class_map_` path (590) and the nested
handler (≈1002); a **holistic** adjuster that resolves *all* `member2t` nodes
post-converter handles every path uniformly, regardless of which converter site
or inference rule built the node. This is the strongest argument for (b) over
(a): one pass, all paths.

**C — Python-only pilot adjuster; the (shared) assert relaxation is safe by (A).
✅** The two pieces have different scopes. The **assert relaxation** lives in
IREP2 core (`irep2_expr.h:1499-1505`/`:1576`) and is unavoidably shared — but
(A) proves it is a no-op for C/C++/CUDA/Solidity (they never build `member2t`
pre-adjust). The **adjuster** can be **Python-only first**: a new IREP2-native
pass replacing the `clang_cpp_adjust` round-trip on Python output only,
generalised to a shared IREP2-native adjust later (RV3). **Recommended first
cut:** (i) relax the two asserts to permit a transient `symbol_type2t` source
(guarded so Release/`NDEBUG` is unaffected and the post-adjust verification is
the real gate); (ii) build the Python-only IREP2-native adjuster against the
20-test fixture; (iii) keep the legacy `clang_cpp_adjust` path for all other
frontends until a later shared phase. Blast radius of the first cut: the two
assert lines + the new Python adjuster + Phase V.3's converter flip — **not** the
shared adjust, not symex.

**Net:** all three investigations come back favourable. The one core change (the
assert) is proven low-risk (A); the adjuster needs no inference of its own (B);
and the first cut is cleanly Python-scoped (C). V.1k is ready to move from
spike to a built **(b)** pilot, gated by the 20-test fixture.

#### V.1k breakthrough — the separate adjuster is **not needed pre-V.4**; the relaxed assert alone unblocks V.3

> Recorded in the §15.1 / §16.1 honesty spirit: building the step-1 relaxation
> exposed that a large part of the design-(b) scope above is **unnecessary at the
> V.3 stage**. The (b) adjuster is real, but it belongs to **V.4+**, not V.1k.

While starting the (b) pilot, an architectural check changed the plan:
`clang_cpp_adjust` operates on **legacy `codet`** symbol values
(`adjust_code(codet&)`), and `goto_convert` migrates to IREP2 **after** adjust
(`goto_convert_functions.cpp:116`). So a *separate* IREP2-native adjuster has no
IREP2 to operate on until function **bodies** are IREP2 — which is **V.4**
(removes P1/W1). A separate adjuster therefore **cannot run pre-V.4**.

That prompted a simpler hypothesis, **confirmed empirically**: with the step-1
assert relaxation, the converter builds `member2t` directly over the
(symbol-typed) base permitted by the relaxation and **back-migrates once** — and
because `migrate_expr_back(member2tc(migrate_type(t), migrate_expr(base), name))`
is the **exact IREP2 round-trip** of `member_exprt(base, name, t)`, the legacy
node handed to the body is **byte-identical**. The **existing**
`clang_cpp_adjust` + `goto_convert` then resolves the symbol source exactly as
today. No converter-side `ns.follow`, **no separate adjuster**.

**Evidence.** Migrating four `member_exprt` sites in `converter_expr.cpp`
(attribute / optional / enum / struct access) to this round-trip form passed
**536/536** attr/class/nested/dataclass/enum/optional/tuple/math tests on an
asserts build — including every family the design-(a) `ns.follow` prototype
regressed (round 2). The contrast is the whole lesson:

| Approach | What it does to the type | Result |
|---|---|---|
| (a) `ns.follow` in converter | **changes** the base type (early resolution) → diverges from adjust | 20/488 regressions |
| round-trip (this) | **changes nothing** — exact IREP2 round-trip of the legacy node | 0 regressions |

**The §16.1 wall was only the construction assert.** Once step 1 removes it,
member/index construction in the converter is just the same
build-IREP2-then-back-migrate pattern file 1 used for non-member expressions
(#5041/#5045) — behaviour-preserving, with resolution left exactly where it is.

**Plan correction (supersedes the (b) scope above for the V.3 horizon):**
- **V.3 (converter builds IREP2 member/index)** is unblocked by **step 1 alone**
  (the relaxed assert) + the round-trip pattern. It does **not** need the (b)
  adjuster. First V.3 member commit: `converter_expr.cpp` 4 sites, behaviour-
  preserving (536/536).
- **The (b) IREP2-native adjuster moves to V.4+**: it is required only once
  bodies are IREP2 and resolution must move off the legacy `clang_cpp_adjust`
  pass (then it also retires W3). The 20-test fixture remains its acceptance
  set *at that point*; pre-V.4 those tests are already green via the round-trip.
- **Net simplification:** V.1k's deliverable is the **assert relaxation**, not an
  adjuster. The keystone was smaller than the (b) scope assumed.

#### V.1k (b)-adjuster — execution scoping (post-V.4.4b: the precondition is now met)

> **Status: B.0 + B.1 + B.2 landed; B.3 spike done (service design, B.0–B.2 → B.5); RV2 layer-1 pinned; RV2 layer-2 gate turnkey + Bitwuzla baseline clean (0/69); B.3 first site (BoolOp cast) LANDED behind the flag, Bitwuzla parity 0-divergence over fixture+broad+boolop, Z3 via CI; B.4 triage done — F-P11 residue substantially stale (member-arith already-resolved), most of B.4 is cheap flag-gated round-trips not follow()-service work (2026-06-26).** The V.1k breakthrough above deferred
> the (b) IREP2-native adjuster to "V.4+" on the grounds that *a separate adjuster
> has no IREP2 to operate on until function bodies are IREP2*. **That precondition is
> now satisfied:** V.4.4b landed and the IREP2 body round-trip is the only body path
> (`goto_convert_functions.cpp:184`, `migrate_expr_back(symbol.get_value2())`). The
> assert-relaxation half of V.1k is also landed (the `symbol_id` disjunct in
> `member2t`/`index2t`, `irep2_expr.h:1553-1556`/`:1637-1638`, documented there as
> "staged enabling infra, exercised once the V.1k converter/adjuster pilot lands").
> So the (b) adjuster is **now buildable**; this section scopes it. It is the single
> remaining unlock for the V.3 residue (the F-P11 general-operand wall) **and** W3
> (`#cpp_type` attribute carriage) — they retire together.

**Why this is the next task (and the V.3 micro-stragglers are not).** The clean,
behaviour-preserving V.3 surface — synthetic-operand subtrees and resolved-source
member/index/deref round-trips — is drained across the whole frontend (converter,
OM handlers, numpy, string/dict/set/tuple, class/lambda/builtins). Every *remaining*
legacy expression-builder site is exactly one of: (i) **F-P11** — an operand resolved
only by `clang_cpp_adjust`, so building IREP2 pre-adjust aborts on the (now-relaxed but
still real, post-adjust) `member2t`/`index2t` resolved-source invariant; (ii) a
**width-hazard** guard/arith the legacy node tolerates but `lessthan2t`/`sub2t` reject;
or (iii) a **statement-context** `code_function_callt`/`code_assignt` operand that is
intentionally legacy until V.4/W1 (e.g. `converter_unop.cpp` `__ESBMC_list_size`
truthiness, commented "the size_decl/size_call statements above stay legacy (P1)").
None of (i)–(iii) is a mechanical migration; all three need resolve-then-build, i.e.
the (b) adjuster.

**What the adjuster is (grounded restatement).** A new IREP2-native pass that runs
the *complete* resolution `clang_cpp_adjust` does today — recursive `symbol_type2t` →
followed `struct_type2t`/`array_type2t` following, pointer auto-deref, dataclass/
inference-aware completion, and `#cpp_type`/`#member_name` carriage — but over the
IREP2 body (`symbol.get_value2()`) instead of the legacy `codet`. It replaces the
legacy `clang_cpp_adjust adjuster(context); adjuster.adjust()` round-trip on Python
output (`python_language.cpp:249-250`). The converter's already-computed inferred
attribute types ride the node as `gen_pointer_type(symbol_typet("tag-<cls>"))`
(`converter_class.cpp:291`, fed by `infer_attr_type_from_usage`/`flow_class_map_`), so
the adjuster *follows* what inference produced — it does not re-infer (V.1k
investigation B).

**Two-phase source invariant (the core mechanism, half-landed).**
1. *Relax* the `member2t`/`index2t` construction assert to permit a transient
   `symbol_type2t` source — **DONE** (`irep2_expr.h`, V.1k step 1). Investigation A
   proved this is a no-op for C/C++/CUDA/Solidity (no frontend builds `member2t`
   pre-adjust; all go through `migrate` at goto-convert, which is post-adjust), so the
   blast radius is exactly the new Python pre-adjust window the adjuster closes.
2. *Re-enforce* the strong invariant post-adjust: the adjuster asserts at exit that
   every `member2t`/`index2t` source is a resolved `struct`/`union`/`array`/`vector`
   — **this pass is what V.1k (b) builds.** The ~11 symex/simplifier consumers
   (`to_member2t(...).source_value`, `struct_union_members`, …) need no change; they
   all run strictly post-adjust.

**Acceptance fixture (already in tree).** The 20-test regression set the design-(a)
`ns.follow` prototype regressed is the acceptance gate — four families, all present
(`regression/python/{nested-attr*,github_4117*,dataclass*,self_ref_nested_attr_chain*}`,
69 tests in those families today). The adjuster must take all four back to green while
keeping the flat-member improvement, because they exercise exactly the resolution the
single-leaf-follow (a) missed: recursive following (`self.b.a`), inferred-type
following (`next: None`→`Node`), and dataclass field resolution.

**Phased, commit-sized plan (gate each on the §V.5 universal gate + `esbmc-cpp`).**
- **B.0 — pass skeleton, dead-but-tested. LANDED.** Added the Python-only IREP2
  walker `src/python-frontend/python_adjust.{h,cpp}`, modelled on
  `clang_c_adjust::adjust()`: it snapshots the symbol list, skips `is_type`
  symbols, and walks each remaining `symbol.get_value2()` via a `const`
  `adjust_expr(const expr2tc&)` that descends every operand through
  `foreach_operand` and **mutates nothing** (the `const` signature makes the
  no-op a compile-time guarantee). Not wired into `python_language.cpp`. The
  contracts — visits-every-node and the `is_type`-skip no-op — are pinned by
  `unit/python-frontend/python_adjust_test.cpp`. Parity is trivially unchanged
  (the pass is on no execution path).
- **B.1 — member/index source following. LANDED.** `python_adjust::adjust_expr`
  is now a mutating post-order walk: it resolves operands first (resolve `X`
  before `X.a`), then rewrites a `member2t`/`index2t` source whose type is an
  unresolved `symbol_type2t` to carry the followed `struct_type2t`/`array_type2t`
  (`namespacet::follow` + `expr2t::with_type`). **Pointer auto-deref is not in the
  adjuster:** the `member2t`/`index2t` construction assert forbids a pointer
  source (`irep2_expr.h:1552-1556`/`:1647-1648` permit only struct/union/complex/
  array/vector/`symbol_id`), so a pointer base is dereferenced — to a
  `symbol_type2t`-typed source — *before* the node is built; the adjuster only
  ever sees the by-name source. Still not wired into `python_language.cpp` (B.2).
  `python_adjust_test.cpp` pins direct member, direct index, and recursive nested
  resolution.
- **B.2 — wire in behind a flag. LANDED.** Added the `--python-irep2-adjust`
  option (`options.cpp`, IREP2-migration group, default off). `python_languaget::
  typecheck` runs `python_adjust` after the legacy `clang_cpp_adjust` when the flag
  is set; `adjust()` is scoped to Python-mode symbols (V.1k RV-adj4 — the C OM
  bodies stay on the legacy path). Flag off ⇒ byte-identical (the branch is not
  taken); flag on is also byte-identical on today's corpus (the converter does not
  yet emit pre-adjust `symbol_type2t` members, so the pass resolves nothing).
  Regression tests `regression/python/python_irep2_adjust{,_fail}` pin both verdicts
  under the flag; a unit test pins the Python-mode scoping.
- **B.3 — converter emits one F-P11 family pre-adjust. RE-SCOPED by the B.3 spike
  (2026-06-26, below): use the resolve-then-build *service*, not the B.2 pass.** The
  original "build `member2t`/`index2t` over a `symbol_type2t` source and rely on B.2's
  pass to resolve it" cannot work while legacy `clang_cpp_adjust` still runs (the spike
  proves both blockers). The viable B.3 flips the *smallest* F-P11 site (e.g. the
  `BoolOp` short-circuit, `converter_stmt.cpp`) to **resolve the source type via the
  converter's `name_space().follow()` at construction** and build a *resolved-source*
  `member2tc`/`index2tc` (the proven file-1 back-migrate-at-seam recipe; satisfies even
  the strong assert). Acceptance unchanged: the relevant slice of the 20-test fixture
  green, dual-solver, asserts build, **plus** RV2 — the service's resolved type must
  equal `clang_cpp_adjust`'s (corpus round-trip equivalence). See the B.3 spike outcome.
  **First site LANDED behind the flag (2026-06-26).** The `BoolOp` short-circuit's
  `to_bool_condition` general cast (`converter_stmt.cpp`) now, under
  `--python-irep2-adjust`, builds the boolean cast via the IREP2 round-trip
  (`migrate_expr` operand → `typecast2tc` → `migrate_expr_back`, locations re-attached)
  instead of the legacy `typecast_exprt`. **Empirical finding:** the operand reaching
  this site is already resolved by `get_expr`, so `migrate_expr` does **not** hit the
  F-P11 member/index assert here (a probe over the fixture confirmed it) — this site is
  the file-1 already-resolved-operand recipe, not the general unresolved-base case the
  later F-P11 families need. **RV2 — Bitwuzla half discharged:** the `PARITY_FLAG=
  --python-irep2-adjust` sweep is 0 divergences over the 20-test fixture (71) **and** a
  130-test broad slice **and** 39 `and`/`or`-heavy tests. Flag off ⇒ byte-identical;
  the Z3 half runs in CI (`regression/python/boolop_member_attr_adjust{,_fail}` pin the
  flipped path). The remaining F-P11 families (unresolved-base member/index) still need
  the `name_space().follow()` resolution and are B.4.
- **B.4 — generalise + post-adjust verification.** Add the exit assertion (every
  member/index source resolved); migrate the remaining F-P11 families; fold in W3
  (`#cpp_type`/`#member_name` IREP2-native carriage), retiring the legacy attribute
  reads at `clang_cpp_adjust_expr.cpp:464`, `cpp_expr2string.cpp:138-140`,
  `goto2c/expr2c.cpp:169-174`.
  **Exit assertion LANDED (2026-06-26).** `python_adjust::adjust()` now re-enforces
  the strong invariant: after resolving each body it walks the result and, if any
  `member2t`/`index2t` source is still a `symbol_type2t` (a survivor the pass could not
  follow — e.g. an unregistered tag, left untouched by the guard), it `log_error`s
  naming the symbol and returns true (error). This is the post-relaxation re-enforcement
  the relaxed construction assert deferred. It is a pure safety net on the live pipeline
  — `clang_cpp_adjust` resolves every source before the flag-on pass runs, so the
  fixture + broad parity sweep stays **0 divergences** with the check active — and it
  catches a B.5-era resolution bug deterministically. Unit-tested both ways (resolved
  body ⇒ false; unregistered-tag survivor ⇒ true).
  **Second site flipped — `isinstance` NoneType null-check (2026-06-26).** Under the
  flag, `build_isinstance`'s `obj == NULL` None-check (`builtins.cpp`) builds via the
  IREP2 round-trip (`migrate_expr` operands → `equality2tc` → `migrate_expr_back`,
  operand location re-attached) instead of the legacy `exprt("=")`. Probe-confirmed
  already-resolved (no F-P11), no width hazard (equality over same-type pointer
  operands). RV2 Bitwuzla half: **0 divergences** flag on vs off over 60
  isinstance/Optional tests + a 90-test broad slice (both this and the BoolOp flip
  active together). Flag off ⇒ byte-identical; `regression/python/isinstance_none_adjust
  {,_fail}` pin the flipped path for the CI Z3 half.
  **Third site flipped — integer `Add`/`Sub` (`build_binary_expression`, 2026-06-26).**
  The largest F-P11 family. Under the flag, same-type integer `Add`/`Sub` build via
  `python_expr::build_add`/`build_sub` (the existing `add2tc`/`sub2tc` round-trip
  helpers). **Width-hazard handled by a guard:** the flip fires only when
  `lhs.type() == rhs.type() == result` and the type is an integer bitvector, so
  `add2t`/`sub2t`'s width-consistency assert always holds; every width-mismatched,
  float, or other case falls through to the legacy node, byte-identical. RV2 Bitwuzla
  half: **0 divergences** flag on vs off over 65 class/arith tests + a 110-test broad
  slice (all three flips active together). A swap-probe (build_add↔build_sub) confirmed
  the flip actually fires (it flipped the verdict). `regression/python/arith_member_adjust
  {,_fail}` pin the flipped path for CI. The width-mismatch / float arith remain legacy
  (the genuine width-hazard residue, §3640).
  **`Mult` added (2026-06-26).** Same guard, via a new `python_expr::build_mul`
  (`mul2tc`); the integer-bitvector + exact-type-match fence covers `mul2t`'s width
  assert identically. Sweep stays **0 divergences** over 65 class/arith + 90 broad
  (four flips active); swap-probe (`build_mul`→`build_add`) made `m.x * 7 == 42` FAIL,
  confirming it fires. `regression/python/mult_member_adjust{,_fail}` pin it. The
  integer `Add`/`Sub`/`Mult` family is now flipped; `Div`/float (`ieee_*`) and
  width-mismatched arith remain the legacy residue.
  **Ordering comparisons `Lt`/`LtE`/`Gt`/`GtE` flipped (2026-06-26).** Same idiom via
  the existing `build_less_than`/`build_less_equal`/`build_greater_than`/
  `build_greater_equal` helpers. The result is bool but the operands carry their own
  type, so the guard is *operand* match — `lhs.type() == rhs.type()` + integer
  bitvector — which discharges `lessthan2t` et al.'s operand-width assert; `Eq`/`NotEq`,
  float, and width-mismatched comparisons stay legacy. Sweep **0 divergences** over 65
  class/arith + 100 broad (relational flip active); swap-probe (`Lt`→`Gt`) made
  `p.a < p.b` FAIL, confirming it fires. `regression/python/cmp_member_adjust{,_fail}`
  pin it.

  #### B.4 triage (2026-06-26) — the F-P11 residue is substantially STALE; most sites are already-resolved
  > The §3636 F-P11 residue list dates to **2026-06-02** (the Phase-4.4
  > `build_binary_expression` trial that aborted ~277/432). Attribute-type resolution
  > in `get_expr` has improved since, so that list is now an **over-count**. A
  > throwaway probe — `migrate_expr(lhs)` / `migrate_expr(rhs)` unconditionally at
  > `build_binary_expression` (`converter_binop.cpp:1953`), rebuilt — runs **clean**
  > (no `irep2_expr.h:1502/1576` abort) on `class1`/`class2`/`class4`/`class5`
  > (member arithmetic: `self.weight += 1`, `self.age + 1`). The same held for the
  > BoolOp site (B.3, the whole fixture). So the operand reaching these arithmetic /
  > boolean sites is **already resolved**, and they are *not* genuine F-P11 aborts —
  > they migrate via the B.3 round-trip recipe (no `follow()` service needed).
  >
  > **Consequence — B.4 re-scoped.** Most of the §3636 list is now the *cheap*
  > already-resolved case: flag-gated round-trip flips like B.3, fully Bitwuzla-
  > validatable + Z3-via-CI, **no dual-solver-gated `follow()`-service work**. The
  > genuinely-unresolved residue (if any survives) must be re-confirmed by probe per
  > site before assuming it needs `name_space().follow()`. **Next B.4 step:** probe
  > each §3636 site, flag-gate the already-resolved ones (the bulk), and only build the
  > `follow()` service for any site a probe proves still aborts. The arithmetic
  > builders (`build_binary_expression`, `handle_modulo`/`handle_floor_division`) need a
  > per-operator `*2tc` mapping rather than the single `typecast2tc` of B.3, so they are
  > a slightly larger (but still already-resolved) flip than the BoolOp cast.
- **B.5 — flip default, then drop the Python `clang_cpp_adjust` round-trip.** Once the
  full 20-test fixture + whole `regression/python` + model `.py` corpus hold parity on
  both solvers (asserts build) and `esbmc-cpp` is green (RV3 — the relaxed assert and
  any shared carriage touch C++ too), make `--python-irep2-adjust` default-on, then
  remove the legacy adjust hop on Python output. **Per the B.3 spike, B.0–B.2's
  materialised-body `python_adjust` pass is the deliverable that does real work *here*
  — it replaces the legacy hop; it is correctly dead-but-tested + flag-gated until this
  point, because legacy `clang_cpp_adjust` resolves every source before it runs.**

#### B.3 spike outcome (2026-06-26) — settles the §V.1k design question: B.3 uses the resolve-then-build *service*; the materialised-body pass is B.5

> Read-only spike of the smallest F-P11 site (`BoolOp` short-circuit,
> `converter_stmt.cpp`) against the live converter→adjust pipeline. It settles the
> open design question recorded above ("type-resolution *service* the converter
> queries at construction vs. materialising IREP2 bodies pre-adjust"): **the service
> wins for B.3; the materialised-body pass (B.0–B.2) is repositioned to B.5.**

**Finding 1 — the original B.3 ("rely on B.2's pass") cannot work pre-B.5.** Two
independent, code-grounded blockers:
1. *The converter body is legacy.* Each function body is a single legacy `codet`
   stored by `set_value(exprt)` (`converter_funcdef.cpp:1403`). A lone IREP2
   `member2tc` cannot be embedded in a legacy `codet`; making the whole body IREP2 and
   writing `set_value(expr2tc)` is V.4/B.5-class, not a one-site flip.
2. *Legacy adjust shadows the pass.* `python_languaget::typecheck` runs
   `clang_cpp_adjust::adjust()` **unconditionally** (`python_language.cpp:249`), and it
   follows the legacy member/index sources. By the time `python_adjust` reads
   `symbol.get_value2()`, the lazily back-migrated body is already resolved — no
   `symbol_type2t` source survives for it to resolve. So B.2's pass is a structural
   no-op on the live pipeline *regardless of converter changes*, until the legacy hop
   is dropped (B.5). (This is consistent with B.2's flag-on byte-identical result.)

**Finding 2 (enabling) — the service is available today.** The converter exposes
`name_space()` (`python_converter.h:187`), so `ns.follow()` resolution is reachable at
construction. An F-P11 site can therefore follow its `symbol_type2t` base to the
resolved `struct_type2t`/`array_type2t` and build a **resolved-source**
`member2tc`/`index2tc` — which satisfies even the *strong* construction assert (no
relaxation needed) and back-migrates cleanly at the seam (the proven file-1 recipe).
This is exactly how the already-merged "resolved-source" sites (#5586/#5587/#5589)
build, except the source is resolved by the service rather than known-resolved.

**Consequence — re-sequencing (B.0–B.2 are not wasted).** B.0–B.2's `python_adjust`
remains correct and is the **B.5** deliverable: the IREP2-native replacement for the
legacy `clang_cpp_adjust` hop, which only has sources to resolve once that hop is
dropped. It stays dead-but-tested + `--python-irep2-adjust`-gated until B.5. B.3/B.4
proceed independently via the **service** (`ns.follow` at construction).

**The load-bearing risk stays RV2 (critical).** `clang_cpp_adjust` does more than
`ns.follow` — pointer auto-deref, dataclass/inference completion. The service's
resolved type must equal what `clang_cpp_adjust` + goto-convert produce, proven by
corpus round-trip equivalence on an asserts build, before any F-P11 site flips.

#### RV2 discharge — layer 1 (type-following) pinned; layer 2 (auto-deref/completion) gated at the flip (2026-06-26)

> A follow-up read of the resolution mechanism splits RV2 into two layers and
> discharges the first. **The spike's "the service is just `ns.follow`" is too
> coarse** — the empirical mechanism is subtler, which is *why* RV2 is critical.

**Mechanism (empirical, GOTO-dump grounded).**
- `clang_c_adjust::adjust_type` does **not** follow a non-macro symbol type
  (`clang_c_adjust_expr.cpp`: it resolves only `is_macro` type symbols); the
  struct-following for a member base happens elsewhere.
- For a Python instance (`pointer→symbol_typet("tag-Cls")`), `adjust_member`
  inserts a **dereference** whose type is the pointer's `subtype`; the member
  source therefore resolves to a `dereference` of the followed struct. A
  `--goto-functions-only` dump of `p.x` confirms it: `ASSERT p->x == 5` — a
  resolved-struct member over a pointer auto-deref, **not** a raw `symbol_type`
  source surviving to symex.

**Layer 1 — type-following (DISCHARGED).** The service resolves the source type
with the IREP2-native `namespacet::follow(type2tc)`; `clang_cpp_adjust` follows
through the legacy `namespacet::follow(typet)`. These are *two distinct
implementations* (the IREP2 one was added to skip the back-migrate detour on the
hot path). A new unit test
(`unit/python-frontend/python_adjust_test.cpp`, `[rv2]`) pins
`ns.follow(migrate_type(s)) == migrate_type(ns.follow(s))` across the 20-test
fixture's class shapes — simple, nested struct-valued, self-referential
(`*Node`), and pointer-to-class (`*B`). Equivalent across all; a divergence in
either follow can no longer silently break the service.

**Layer 2 — auto-deref + dataclass/inference completion (gated, not yet
discharged).** The pointer auto-deref and dataclass-field completion are *not*
covered by the type-following test. They are validated end-to-end at the first
site flip via the corpus parity sweep (`scripts/irep2-migration/parity_sweep.sh`,
deterministic verdict, dual-solver Bitwuzla + Z3, asserts build) — RV-adj2 mandates
verdict/text parity over a goto diff because of model nondeterminism.

**RV2 layer-2 gate is now turnkey (2026-06-26).** `parity_sweep.sh` was generalised
to gate any staged flag via `PARITY_FLAG` (default `--irep2-bodies`, backward-
compatible). The F-P11 flip is staged behind `--python-irep2-adjust` (default off,
the B.2 flag), so its RV2 gate is a single command:
`PARITY_FLAG=--python-irep2-adjust parity_sweep.sh <esbmc> <fixture>`, comparing
flag-off (legacy) against flag-on (resolve-then-build) — they must agree.
The **20-test fixture** is concretely the
`regression/python/{nested-attr*,github_4117*,dataclass*,self_ref_nested_attr_chain*}`
families (69 test.descs today). **Baseline established:** at the current B.2 state the
sweep is **0 divergences / 69** on Bitwuzla — the no-op pass is parity-clean, the
reference the flip must preserve.

**Environment note (hard).** The mandatory dual-solver gate needs Z3 *and* Bitwuzla.
A Bitwuzla-only build (e.g. local dev here) discharges only the Bitwuzla half; the Z3
half runs in CI on the PR. **Commit policy:** because the flip is behind a default-off
flag, the change is byte-identical by default and safe to land; the flag-on path is
*not* RV2-discharged until the CI Z3 parity run is green.

**B.3 parity anchor pinned (2026-06-26).** The exact BoolOp-over-member-operands
pattern the flip targets is now a CORE regression test
(`regression/python/boolop_member_attr{,_fail}`): a class with `box.a and box.b` /
`box.a or box.b` short-circuits over attribute operands. It verifies SUCCESSFUL /
FAILED identically with the flag **off and on** today — the concrete parity target the
sweep checks; the flip must preserve it. **Next task:** implement the flag-gated flip —
at the BoolOp site, when `--python-irep2-adjust` is set, resolve the member-operand base
type via `name_space().follow()` and build the short-circuit condition as a
*resolved-source* IREP2 node (the file-1 migrate→build→back-migrate recipe), instead of
letting a naive `migrate_expr` of the unresolved operand abort (F-P11). Confirm Bitwuzla
parity over the fixture + this anchor; let CI close the Z3 half before flag-on is
trusted. The change is safe to land regardless (default-off flag).

**Risks (extend §V.4 / Part II §7).** *RV-adj1:* the new pass must reproduce
`clang_cpp_adjust`'s dataclass/inference completion exactly — mitigate by reusing the
converter's already-inferred `tag-` types (investigation B) rather than re-deriving.
*RV-adj2:* GOTO-byte parity vs the legacy adjust path is the gate, but model
nondeterminism (§8.1) means use **deterministic verdict+matched-text parity**, not the
goto diff. *RV-adj3:* W3 carriage is shared with C++ — stage it last (B.4) and gate on
`esbmc-cpp`. *RV-adj4:* keep the pass **Python-only** until a later shared phase;
investigation C confirmed the first cut is cleanly Python-scoped (the relaxed assert is
the only shared change, and it is a no-op for other frontends by investigation A).

**Definition of done (this sub-program).** F-P11 + width-hazard residue migrated; W3
retired; the legacy `clang_cpp_adjust` round-trip is gone from the Python path; the
20-test fixture and full corpus hold dual-solver + asserts parity; `esbmc-cpp` green.
At that point the Python converter writes IREP2 member/index/general-operand
expressions directly, and §V.1 acceptance bars #1/#2 fall for the converter's
expression surface. (Bodies/`goto_convert` W1 stays a RETAIN_BOUNDARY per the V.4
outcome; the C/C++ counterexample printer W4 stays Phase V.5.)

### Phase V.1a — Type construction → `type2tc` end-to-end (extends Phase 4.3)
Finish what Phase 4.3 deferred: the tuple/optional **struct** builders (§15.7
F-P5 seam cases) and any remaining `type_handler` families, now written
straight into the symbol table via `set_type(type2tc)` rather than
`lower_to_seam`. Depends on V.2 (struct components carry `#member_name`/
`#access`, which must by then ride IREP2). *Accept:* `type_handler` constructs
no legacy `typet`; symbol types are IREP2 at write; verdict + text unchanged.

### Phase V.2 — IREP2-native attribute carriage (removes W3)
Move `#cpp_type`/`#member_name`/`#cformat` off `irept` onto a **typed IREP2
companion keyed by symbol id** (the §5.1 pattern — *not* a generic string-map,
which re-introduces the escape hatch IREP2 abolished). Teach the three external
consumers (`clang_cpp_adjust_expr`, `cpp_expr2string`, `goto2c/expr2c`) to read
the IREP2 form. *Accept:* the legacy-attribute reads are gone; verdict + text
unchanged on **both** `python` and `esbmc-cpp` (these consumers serve C++ too).
*Risk:* highest shared blast radius after V.4.

### Phase V.3 — IREP2 expression construction in the converter (Phase 4.4/4.5, for real)
With W2/W3 removed, flip every converter idiom to typed factories — the ~842
`symbol_expr`, member/index/typecast/deref/const, and the RP13 operand surgery
in the OM handlers (`python_list`/`dict`/`set`) — and write `set_value(expr2tc)`.
This is the Part IV §7.2 file-by-file work, now **unblocked**. *Accept:* legacy
expression-builder census in `src/python-frontend` → 0 except body shells;
verdict + text unchanged; asserts cross-check silent. *This is the
"comparatively mechanical" bulk — but only after V.1k.*

#### V.3 status (2026-06-22) — two migratable surfaces (synthetic-operand + guarded member/index round-trip) are now ~drained; general-operand construction stays blocked

Per the **V.1k breakthrough** above, V.1k's keystone — the `member2t`/`index2t`
construction-assert **relaxation** (now permitting a transient
`struct`/`union`/`complex`/`symbol` member source and `array`/`vector`/`symbol`
index source, `irep2_expr.h` ~1553/~1637) — **has landed on master**. The separate
(b) IREP2-native adjuster is *not* needed for V.3; it moves to V.4+. So **two**
converter surfaces are migratable today, and both are now largely drained:

1. **Synthetic-operand subtrees** — constructions every forward-migrated operand of
   which is a fully-synthetic concrete-typed value (OM-call results like
   `__ESBMC_list_size`/`strlen`, `size_type`/`bool` temporaries, constants, already
   -built comparisons). ~25 PRs merged (the `[python] V.3: build … in IREP2` series,
   #5454–#5522): truthiness checks, list/set/tuple/string/complex result-bools,
   is/identity equality, deref/member *sandwich* collapses, chained-comparison
   and-folds, generic unary operators.
2. **Guarded member/index/deref round-trips** — `migrate_expr_back(member2tc/
   index2tc/dereference2tc(migrate_type(t), migrate(base), …))`, which is the
   *byte-identical* round-trip of the legacy node (`member`/`index`/`dereference`
   lower uniformly in `migrate_expr`, `util/migrate.cpp:1580`), guarded on the
   source type (`is_struct/union/symbol` for member, `is_array/vector/symbol` for
   index) with a legacy fallback otherwise. The `build_member`/`build_index`/
   `build_dereference`/`build_typecast`/`build_symbol` helpers across the frontend
   are all migrated this way; the last clean *returned* raw sites — complex
   truthiness (#5533) and `ord()`'s char load (#5534) — followed.

**What is still blocked (the genuine F-P11 residue).** General-operand construction
over *arbitrary user expressions* — `BoolOp` (`and`/`or`), arithmetic in
`build_binary_expression`, and member/index whose source's type is **neither a
resolved struct/array nor a `symbol`** — cannot use the migrate-forward recipe. A
worked attempt to fold the `and`/`or` `BoolOp` node into `and2tc`/`or2tc` (the
sibling of the merged chained-comparison fold #5519) **reproduced an `index2t`
abort** (`irep2_expr.h:1638`) even on pure-boolean `(a>0) and (b>0)`: the operand
carried an unresolved `index` whose source matched *none* of the relaxed disjuncts.
The relaxation widened the migratable surface but did **not** remove the wall for
operands the converter has not yet typed — those still need the full (b) adjuster
(V.4+). Reverted, not shipped.

**Consequence.** Both clean surfaces (synthetic-operand subtrees and guarded
member/index round-trips at *returned/composed* seams) are ~drained in the core
converter. The remaining raw member/index/deref sites all **feed legacy consumers**
(`code_assignt` LHS, `tuple`/`dict_handler` args) — migratable in isolation but
**no net gain** under the subtree rule. The next structural unlock is the **(b)
IREP2-native adjuster at V.4+**, which retires the residual general-operand wall and
W3 together; pre-V.4 the migratable V.3 bulk is essentially complete.

#### V.3 status (2026-06-24) — synthetic-operand + resolved-source member/index/deref surfaces fully drained; residual is exactly F-P11 + width-hazard

A second draining pass (≈16 PRs, #5573–#5595) extended the 2026-06-22 status
above and exhausted every *clean, byte-identical* converter site reachable
pre-V.4:

- **Shared builder module.** The per-TU copies of the `build_*` round-trip
  helpers were consolidated into
  `src/python-frontend/python_expr_builder.{h,cpp}` (`namespace python_expr`,
  #5568), then extended with `build_not`/`build_or`/`build_notequal` (#5573,
  #5575, #5576) for the boolean/equality nodes. The relational/logical/
  arithmetic builders now share two private templates — `migrate_binary` and
  `migrate_typed_binary` (#5594) — so the migrate→build→back-migrate round-trip
  has a single source of truth.
- **Synthetic-operand arithmetic & relational.** List loop-index increments
  (#5584), list `strlen+1` element sizing (#5580), starred-unpack index
  subtraction (#5578), the string-index OOB bound `array_size-1` (#5581), `set`
  char* pointer arithmetic + dereference (#5582, #5590), the `input()`
  length-bound assume (#5583), `str.startswith`/`endswith` length/offset
  arithmetic (#5574, #5595), and the affix-tuple `or`-chain (#5575) — all moved
  to the shared helpers, each a byte-identical round-trip over operands the
  converter has already typed `size_type`/`bool`.
- **Resolved-source member/index/deref feeding legacy consumers.** Sites whose
  source is a *verified resolved* struct/array (not the F-P11 user case): the
  list symbolic-size store `*(list).size` (#5586), the constructor default-attr
  store `*(self).attr` (#5587), and the `str(char)` 2-element array subscripts
  (#5589), via `build_dereference`/`build_member`/`build_index`. These reduce
  the legacy expr-builder census but do *not* remove the migrate seam (subtree
  rule).

**Residual — provably blocked, enumerated.** What remains in the converter is
*exactly* two permanently-blocked classes (neither is a mechanical migration):

1. **F-P11 user-operand sites** — operands resolved only by `clang_cpp_adjust`,
   so building IREP2 pre-adjust aborts on the `member2t`/`index2t`
   resolved-source assert: `python_math.cpp` modulo/floor-div sign-correction,
   `BoolOp` short-circuit (`converter_stmt.cpp`), `isinstance` NoneType/tuple
   (`builtins.cpp:369/447`), is-none inequality (`converter_compare.cpp`),
   slice-bound arithmetic (`python_list.cpp:1444-1683`), the complex int→double
   promotion (`numpy_call_expr.cpp:1607`), and the converter subscript-base
   derefs (`converter_expr.cpp:1369/1388`).
2. **Width-hazard guards/arithmetic** — a `<`/`-` whose two operands have
   *different* widths (a `size_type` index vs an `arr_type.size()`-typed length),
   which the legacy node tolerates (adjust reconciles) but `lessthan2t`/`sub2t`
   reject at construction: `python_set.cpp:171/234`, `python_list.cpp:4120/4533`.
   (`build_array` types its dimension `size_type`, but the length operand is not
   guaranteed to originate there — #5581 defensively typecasts before
   subtracting, so a blind migrate is unsound.)

Both classes need the **V.1k/V.4 IREP2-native adjuster** (resolve-then-build),
not site-by-site migration.

#### V.3 follow-up (2026-06-24) — truthiness/membership stragglers drained

A re-census after the pass above found the "fully drained" claim was slightly
overstated: five clean synthetic-operand sites — each a *byte-identical
round-trip* analog of an already-merged migration — were still building legacy
`exprt("notequal"/…)` nodes. All were drained to the
`migrate_expr_back(notequal2tc/equality2tc(…))` pattern:

- `converter_stmt.cpp` list-condition `__ESBMC_list_size(xs) != 0` (analog of the
  already-migrated truthiness path at the same file ~3216);
- `converter_stmt.cpp` dict-truthiness `$dict_size$ != 0`;
- `string_handler.cpp` `strchr(...) != NULL` and `strstr(...) != NULL` membership
  (analog of the merged `strcmp(...) op 0` at `converter_binop.cpp:973`);
- `python_exception_handler.cpp` non-negated assertion `(int)<bool-temp> == 1`
  (sibling of the already-migrated `not <temp>` branch).

Operands are all fully-synthetic concrete-typed (OM-call results, `size_type`
temps, constants), so each is the exact IREP2 round-trip of the legacy node —
verdict + counterexample parity holds (dual-solver Bitwuzla + Z3). With these,
the clean V.3 surface is drained; the residual is exactly the two
permanently-blocked classes above.

#### V.3 residual → the remaining tasks to reach 100 % IREP2 (2026-06-24)

The pre-V.4 clean V.3 surface is drained; the residual is exactly the two
classes above (F-P11 user-operand sites + width-hazard guards). Both share one
root cause and therefore one fix, so the path to **100 % IREP2** is a small,
ordered task list — not an open-ended sweep. A read-only spike pins the
constraint the keystone task must respect:

- **The resolution lives in a legacy-`codet` adjust pass.** `python_languaget::
  final` runs `clang_cpp_adjust::adjust()` over the symbol values
  (`python_language.cpp:249-250`, entry `adjust_code(codet&)`,
  `clang_cpp_adjust.h:32`); this is where the converter's unresolved
  `member`/`index` sources get followed (the P2/W2 resolution). The residual
  aborts because the converter builds those nodes **before** this pass.
- **The goto-convert body round-trip stays legacy (W1-loc) — and is off this
  critical path.** V.4 concluded the *body* round-trip is a `RETAIN_BOUNDARY`
  for source-location fidelity. That governs `goto_convert`'s side-effect
  hoisting, **not** the converter→adjust type-resolution seam where the residual
  is built, so it does not block the tasks below.

**Tasks to 100 % (ordered; the first is the keystone, RV2 the gate):**

1. **V.1k — resolve-then-build via the design-(b) IREP2-native adjuster.**
   Stand up the Python-specific IREP2 adjuster that performs `clang_cpp_adjust`'s
   *complete* resolution (recursive struct-following + auto-deref +
   dataclass/inference-aware completion) as one IREP2 pass, so the converter can
   build `member2tc`/`index2tc`/arith with **resolved** operand types and the
   F-P11 + width-hazard aborts vanish. **Open design question to settle in the
   first spike:** whether this runs as a type-resolution *service* the converter
   queries at construction (no materialised IREP2 body needed) or requires
   materialising IREP2 bodies pre-adjust (the resolved-source chicken-and-egg
   the relaxed assert only partially lifts). *Accept (RV2, hard gate):* the
   20-test acceptance fixture (§V.1k design (b)) back to green; the F-P11/
   width-hazard sites build in-converter with zero `irep2_expr.h` aborts;
   verdict + text unchanged.
2. **V.2 — IREP2-native attribute carriage (W3).** Fold `#cpp_type`/
   `#member_name`/`#cformat` onto the typed IREP2 companion; the (b) adjuster is
   the natural home, so V.1k and V.2 land together.
3. **V.5 — IREP2-native counterexample printer (W4).** Independent of the above
   (Part II §2.7, own issue); can proceed in parallel.
4. **V.6 — flip & verify.** Switch the symbol-table writes fully to
   `set_value(expr2tc)`, census → ~0, dual-solver + asserts + `esbmc-cpp`
   parity. §V.1 bar met.

Converter-side V.3 *construction* is drained as far as the pre-adjust assert
allows; everything remaining is the V.1k keystone and what it unblocks. The
deeper W1 body round-trip (V.4) is the one piece retained by design — it is a
shared goto-convert concern, not part of the Python 100 % target.

### Phase V.4 — IREP2 structured CF + IREP2-aware `goto_convert` (removes W1)
Add the missing structured CF code kinds to IREP2 (`code_ifthenelse2t`,
`code_while2t`, `code_for2t`, `code_switch2t`, `code_break2t`,
`code_continue2t`, `code_label2t`) and teach `goto_convert` /
`goto_convert_functions` to consume IREP2 bodies (or migrate at a thinner,
IREP2-side seam). The converter then emits IREP2 bodies and
`goto_convert_functions.cpp:116` no longer back-migrates. *Accept:* a Python
body round-trips as IREP2 through goto-convert with **byte-identical GOTO
output**; **all** frontends still pass. *The biggest, riskiest phase — it
changes every frontend's body lowering; stage behind a feature flag and migrate
frontends one at a time.*

#### V.4 grounding — the current legacy-body seam (read-only investigation, 2026-06-09)

The body is consumed legacy, **per-instruction**, never migrated whole:

- **Entry:** `goto_convert_functionst::convert_function`
  (`goto_convert_functions.cpp:97-201`): line 117 migrates the *function
  type* (`migrate_symbol_type`), line 129 extracts the body as legacy
  `const codet &code = to_code(symbol.get_value())`, line 152 dispatches
  `goto_convert_rec(code, f.body)`. The body itself is **not** migrated.
- **Dispatch:** `goto_convertt::convert(const codet&, goto_programt&)`
  (`goto_convert.cpp:220-296`) switches on `code.get_statement()` (string
  ids) to ~30 handlers. The structured-CF entries are: `"ifthenelse"` →
  `convert_ifthenelse` (1712), `"while"` → `convert_while` (1182),
  `"dowhile"` → `convert_dowhile`, `"for"` → `convert_for` (1082),
  `"switch"` → `convert_switch` (1343), `"switch_case"` →
  `convert_switch_case`, `"break"` → `convert_break` (1426), `"continue"`
  → `convert_continue` (1497), `"label"` → `convert_label` (140),
  `"block"` → `convert_block`.
- **Where legacy becomes IREP2:** each handler reads the legacy sub-parts
  (`to_code_ifthenelse(c).cond()/then_case()/else_case()`,
  `to_code_while(c).cond()/body()`, `code_fort::init/cond/iter/body`,
  `code_switcht::value/body`, `code_labelt::get_label/code`) and calls
  `migrate_expr` on the **expression** operands at instruction-build time
  (e.g. guards at 1151/1407/1798, return at 1480). So migration is
  scattered across the handlers, not a single pre-pass.

→ **Consequence for V.4:** the body-seam can flip in either of two shapes
— (i) make the converter emit IREP2 bodies and rewrite each
`convert_*` handler to read the `code_*2t` fields directly (drops the
per-handler `migrate_expr`), or (ii) keep the handlers but add an
IREP2→legacy `migrate_expr_back` shim at the entry so the converter can
emit IREP2 while goto_convert stays legacy internally. (i) is the real
win (removes W1); (ii) is a thinner intermediate that still pays one
round-trip. Decide per the byte-identical-GOTO gate.

#### V.4 commit sequence (dead-but-tested first — the V-track pattern)

Ordered lowest-risk first; one reviewable commit each; gate every commit
on the full unit suite + `regression/{python,esbmc,esbmc-cpp,floats}`
verdict parity, dual-solver, asserts build (§V.5).

1. **V.4.0 — code-kind infrastructure, dead-but-tested.** **LANDED
   [#5265](https://github.com/esbmc/esbmc/pull/5265).** Add the 7
   structured-CF kinds to `expr_kinds.inc` + `irep_typedefs(...)` +
   class defs in `irep2_expr.h` + `field_names` in `irep2_expr.cpp`; add
   `migrate_expr` (forward, legacy `code_*t` → `code_*2t`) and
   `migrate_expr_back` arms in `migrate.cpp`; add round-trip unit tests
   (`unit/util/migrate.test.cpp`: `migrate_expr_back(migrate_expr(c))==c`
   per kind). **No pipeline wiring** — nothing builds these yet, so the
   commit is behaviour-inert by construction (mirrors Part I V1 #4737 and
   Phase 4.2 #4997). Recipe per kind: see the §18-adjacent memo; each is
   ~1 line manifest + 1 line typedef + ~12-line class + 1 line names + 2
   migrate arms + 1 test. `num_type_fields = 6` accommodates `code_for2t`
   (init/cond/iter/body). The hand-written expr-id switches all carry a
   `default:`, so the additions do not break `-Werror`.
2. **V.4.1 — source location carriage, dead-but-tested.** **LANDED
   [#5266](https://github.com/esbmc/esbmc/pull/5266).** Give the 7
   structured-CF kinds a `locationt location` member (non-reflected —
   outside the `fields` tuple and excluded from cmp/crc/hash via
   `K::excluded_field_bytes`). `migrate_expr` copies `code.location()`
   into the field; `migrate_expr_back` restores it. Adds a round-trip
   test asserting the location survives (it is outside `operator==`, so
   the existing `==` round-trip cannot catch a drop). Prerequisite for
   V.4.2: `goto_convert` reads `code.location()` at ~15 sites; without
   this, IREP2 bodies would lose source locations in the goto output.
3. **V.4.2 — flag + IREP2-side `goto_convert` entry.** **LANDED (#5277)**
   `--irep2-bodies` (default off) routes `convert_function` through
   an IREP2 body round-trip: `migrate_expr` the legacy body to
   `code_*2t`, then `migrate_expr_back` to `codet`, then process through
   the existing `goto_convert_rec` handlers. Flag off ⇒ byte-identical to
   today. New migrate arms: `sideeffect_assign2t` (covers all 13 assign/
   compound-assign variants), `code_switch_case2t`, 2-op `code_decl`,
   `decl-block`. Fixed a latent UB in `ifthenelse` migration (2-operand
   form from Clang C frontend). Regression tests
   `github_4715_irep2_bodies_01{,_fail}` gate on `--irep2-bodies`.
4. **V.4.3 — one frontend at a time.** **LANDED (all frontends).**
   - **Python** (#5281, commit `45a024710c`): added `code_dowhile2t`,
     `code_assert2t`, `code_assume2t` IREP2 kinds; `sideeffect("cpp-throw")`
     forward/back migration; tests `github_4715_irep2_bodies_py_01{,_fail}`.
   - **C** (#5282, commit `8deb03d108`): extended `code_decl2t` with optional
     `init` field; 2-op `code_decl(sym, init)` preserved directly (was
     incorrectly split into `code_block` causing premature DEAD);
     `code("decl-block")` children flattened inline (prevents spurious extra
     scope boundary); `sideeffect("statement_expression")` (GNU `{ }` /
     `assert()` macro) added to `sideeffect_allockind` with forward/back arms;
     tests `github_4715_irep2_bodies_c_01{,_fail}`.
   - **C++** (#5284) and **Jimple + Solidity** (#5286) followed, each its own
     commit; CUDA rides the C/clang-c path.
   - **Exception round-trip fixes (post-flip).** A verdict-parity sweep over
     the flag found two exception-handling defects that the per-frontend
     `_01` tests missed — both **flag-only**, the legacy (flag-off) path is
     unaffected:
     1. *throw dropped → false `SUCCESSFUL`.* A `side_effect_exprt("cpp-throw")`
        nested in a `code_expression` round-trips to the code form
        `codet("cpp-throw")`; `convert_expression`'s `remove_sideeffects`
        recognizes only the side-effect form, so the throw became an inert
        `OTHER`. Affected **every** frontend that throws (Python `raise`/builtin
        TypeErrors, C++ `throw`). Fix: `convert_expression` dispatches a
        code-valued operand via `convert(to_code(...))` (goto_convert.cpp).
     2. *try/catch → SIGSEGV.* `code_cpp_catch2t` stored only `exception_list`
        and **no operand storage**, so the forward migrate dropped the try/
        handler blocks and `convert_catch` read `op0()` on an empty operand
        list. Fix: added `std::vector<expr2tc> operands` to the kind; forward/
        back migration carry the operands and re-attach each handler's
        `exception_id` from the parallel `exception_list`; the post-goto-convert
        CATCH marker (no operands) is preserved by disambiguating on
        `operands.empty()` (irep2_expr.{h,cpp}, migrate.cpp).
     Gated dual-solver (Z3 + Bitwuzla) with the asserts cross-check live; tests
     `github_4715_irep2_bodies_{py,cpp}_exc_01{,_fail}`.
5. **V.4.4 — flip the default, then remove the legacy path.**
   - **V.4.4a — flip the default. LANDED (this PR).** `--irep2-bodies` now
     defaults on (`esbmc_parseoptions.cpp`, right after `options.cmdline`);
     `--no-irep2-bodies` is the new escape hatch back to the legacy path.
     Justified by the deterministic verdict-equivalence sweep
     (`scripts/irep2-migration/parity_sweep.sh`) run flag-on across **all**
     frontends — **0 divergences over 8 236 tests** (Python 3 763, C 1 313,
     floats 92, cstd 128, CUDA 131, esbmc-cpp 2 091, cpp11/14/17/20 214,
     Solidity 504). The matched-text gate (ctest) then surfaced three
     location-text divergences (`switch-line-no`, `print-function-deps_{1,3}`)
     that the verdict-only sweep cannot see; these were already fixed upstream
     by **#5348** (value-operand location back-fill) and **#5357**, both of
     which this work rebased onto before re-validating green. Tests
     `github_4715_no_irep2_bodies_01{,_fail}` pin the escape hatch.
     *Caveat:* the verdict sweep ignores boolector-only `THOROUGH` tests in a
     Z3-only build (pre-existing environment failures, identical on both
     paths).
   - **V.4.4b — remove the legacy path. LANDED.** `convert_function`
     (`goto_convert_functions.cpp`) now lowers every body through the IREP2
     round-trip unconditionally: the `to_code(symbol.get_value())` legacy
     bypass, the `options.get_bool_option("irep2-bodies")` gate, the
     `esbmc_parseoptions.cpp` default-set, and the `--no-irep2-bodies` escape
     hatch are all deleted — the IREP2 round-trip is the only body path.
     `--irep2-bodies` is retained as a deprecated, accepted **no-op** so the
     49 `github_4715_irep2_bodies_*` regression tests (and any external
     scripts) keep working without a mass `test.desc` rewrite; those tests now
     validate the unconditional path. Behaviour is identical-by-construction to
     V.4.4a's already-validated default-on path (this commit only removes the
     now-unreachable else-branch and the flag plumbing). The exception fixes
     above were the first parity blockers found; #5348/#5357 and the
     per-construct parity fixes (#5330/#5335/#5340/#5346/#5355) closed the
     rest, since the per-frontend `_01` smoke tests do not exercise every body
     construct. *Next:* the deeper W1 removal — rewrite the `goto_convert_rec`
     handlers to consume `code_*2t` directly — was scoped and prototyped and is
     a **closed boundary** (location-fidelity wall); see *V.4 outcome* below.

**Risk/scope:** V.4.0 and V.4.1 are small and safe (infra only). V.4.2+
touch the shared goto pipeline → gate on `esbmc-cpp` and a
Solidity/CUDA stratum, not only `python` (RV3), and require
byte-identical GOTO (RV4). Stage behind the flag; never flip two
frontends in one commit.

#### V.4 outcome — the deeper W1 removal is a closed boundary (location-fidelity wall)

> **Status: concluded (2026-06-19).** V.4.0→V.4.4b landed (the IREP2 round-trip
> is the only body path). The *deeper* W1 step §V.4.4b flagged as "Next" —
> rewriting `goto_convert_rec` to consume `code_*2t` **directly**, dropping the
> back-migration — was scoped and prototyped, and the conclusion is that it is
> **not achievable as "remove the round-trip."** The body round-trip to legacy
> is fundamental for **source-location fidelity**, not merely side-effect
> cleaning. Recorded here so it is not re-attempted without new infrastructure.

**The two walls, both proven empirically.**

1. **W1-clean — cleaning must be total, and is legacy.** (Established earlier:
   `goto_sideeffects.cpp` is 100 % `exprt`; `goto_symext::handle_sideeffect`
   `assert(0)`s on any non-allocation side-effect, so `function_call`/`++`/`--`/
   comma must be **fully hoisted** at `goto_convert`.) The W1 dual-API seam
   (`remove_sideeffects(expr2tc&)`, [#5414](https://github.com/esbmc/esbmc/pull/5414))
   plus its native `has_sideeffect`/fast-path
   ([#5417](https://github.com/esbmc/esbmc/pull/5417)) are landed, but the body
   of the hoisting is still legacy.

2. **W1-loc — value-operand source locations cannot live in IREP2.** This is the
   decisive wall. IREP2 value nodes (`symbol2t`, `constant_int2t`, …) carry **no
   per-node location** — the closed type system omits it by design (the same
   value-traveling attribute the migration set out to remove; cf. Part III
   `#sol_*`, Part IV `#cpp_type`). `goto_convert`'s side-effect removal
   *generates* instructions (temp decls/assignments for calls-in-expressions,
   guard checks) whose locations come from the **value operands'** source
   locations. The body round-trip → legacy → `restore_value_locations`
   ([#5348](https://github.com/esbmc/esbmc/pull/5348),
   `goto_convert_functions.cpp`) is the mechanism that supplies them.
   **Proof:** no-op `restore_value_locations` and the generated instructions for
   `int y = f() + 1;` lose their location (`line 3 column 3` → blank). There is
   no IREP2-native equivalent without re-adding per-node locations to the value
   nodes — rejected, for the same reason every other value-traveling-metadata
   wall in this document was.

**The favourable sub-slice was prototyped and measured.** The only statements
that carry *no* value-operand locations are the structural leaves
(`skip`/`goto`/`label`/`break`/`continue`), which could in principle be consumed
natively. Instrumenting `goto_convert`'s dispatcher over 25 `regression/esbmc`
programs: **skip/goto are 3.30 % of statements** (812 / 24 611). And 3.3 % is a
*count* ceiling that overstates the benefit three ways:

- skip/goto carry the **smallest** payloads (a `skip` is empty, a `goto` is one
  guard), so the migrate *work* saved is far below 3.3 %;
- consuming them natively requires the **whole** body to run through an
  IREP2-native dispatcher, so every value statement is then migrated back
  **per-statement** instead of via today's single whole-body `migrate_expr_back`
  — net migrate work ≈ unchanged, minus the trivial skip/goto subtrees;
- it still doesn't remove the round-trip for value statements (W1-loc).

The cost to capture that negligible, mostly-*relocated* saving is a new IREP2
dispatcher + a reimplementation of `convert_block`'s destructor-stack /
`end_location`-unwind logic on `code_block2t` + goto target-tracking + flag
plumbing + **byte-identical-GOTO validation across every frontend**. Poor trade;
**not pursued.**

**What W1 did deliver (kept):** the dual-API `remove_sideeffects(expr2tc&)` seam
and native `has_sideeffect` (#5414/#5417 — the IREP2 entry future cleaning work
hangs off), and `code_block2t` `end_location` carriage
([#5426](https://github.com/esbmc/esbmc/pull/5426)) — which, beyond being the
prerequisite that exposed the W1-loc wall, **fixed a latent regression**: the
V.4.4b round-trip was dropping nested-block `end_location`s, leaving nested-scope
C++ destructor instructions unlocated.

**Conclusion.** The body round-trip stays, by the same governing rule as the
frontend boundaries: the legacy form carries value-traveling metadata
(per-node source locations) that IREP2's closed type system deliberately does
not, and verification-visible fidelity (counterexample text, witness columns,
coverage) depends on it. W1 is therefore a **RETAIN_BOUNDARY**, not a pending
migration. Re-opening it requires either an IREP2-native location-carriage
design (a large, separate initiative) or a decision that location fidelity may
degrade (it may not — it is `test.desc`- and witness-visible).

### Phase V.5 — IREP2-native counterexample printer (removes W4)
Part II Phase 2.7: an IREP2 C/C++ printer so traces / `test.desc`-matched text
are produced without the legacy printer reading `#cpp_type`/`#cformat`.
*Accept:* counterexample text byte-identical where `test.desc` pins it (Q-P1:
the `index < l->size` gate and the coverage/diagnostic families), with the
legacy printer out of the Python path.

### Phase V.6 — Flip & verify
Switch `create_symbol`/`add_symbol`/`update_symbol` fully to IREP2 writes;
delete residual legacy construction and now-dead includes (C-Dead discharge per
removed branch); census → ~0; whole-suite dual-solver + asserts + `esbmc-cpp`
parity. *Accept:* §V.1 bar met; verdict + text parity; go/stop recorded with
the final surviving boundary-glue enumerated.

## V.3 Dependencies

```
V.0 ─► V.1k ─┬─► V.1a ─┐
             ├─► V.2 ──┼─► V.3 ─► V.6
             │         │
   (parallel shared)   │
        V.4 ───────────┤   (bodies + goto-convert)
        V.5 ───────────┘   (text parity)
```
V.1k is the unlock for all converter construction (V.1a/V.2/V.3). V.4 and V.5
are parallel shared tracks that also gate the final flip (bodies, text). V.6 is
the last step.

## V.4 Risk register (Part V-specific; extends Part IV §9)

| # | Area | Risk | Sev | Mitigation |
|---|---|---|---|---|
| RV1 | scope/process | Reopens B1/P1/P2 — boundaries Parts I & III closed as shared and risky. Multi-quarter, multi-engineer, dozens of PRs across all frontends. | **critical** | Run as its own umbrella issue, **not** under Part IV; feature-flag V.4; migrate frontends one at a time. |
| RV2 | soundness | V.1k's resolved type ≠ what `adjust`+goto-convert produce → silent verdict drift on attribute/subscript-heavy programs. | **critical** | The V.1k proof obligation (corpus round-trip equivalence, asserts-on) is a hard gate; do not proceed to V.3 until discharged. |
| RV3 | soundness | V.2/V.4 touch shared clang-cpp/goto passes → C++/CUDA/Solidity verdict regressions, not just Python. | high | Gate **every** V.2/V.4 commit on `esbmc-cpp` + a Solidity stratum, not only `python`. |
| RV4 | compatibility | V.4 changes GOTO-binary lowering → on-disk `.goto` format drift (Part I B4). | high | Require byte-identical GOTO output per phase; pin the serialized format; old binaries must still load. |
| RV5 | text parity | V.5 printer divergence breaks `test.desc`-matched counterexample text. | med | Q-P1 gate (§15.6): the one `index < l->size` pretty-print + coverage/diagnostic families held invariant. |

## V.5 Validation

Part IV §10 gate, escalated: dual-solver (Bitwuzla + Z3) verdict + matched-text
parity, asserts build (live `migrate.cpp` cross-check), over the full
`regression/python` strata **and** the model `.py` corpus — **plus** `esbmc-cpp`
(and a Solidity/CUDA stratum) for any phase that touches shared passes (V.2,
V.4, V.5). GOTO-output byte-identity is an additional V.4 gate. Respect the
5-minute per-run cap by stratifying; full suite at phase boundaries.

## V.6 Honest estimate and recommendation

**Estimate.** This is effectively *completing* the repo-wide IREP2 migration
(#4715) through `goto_convert` and `clang_cpp_adjust` — **multi-quarter,
multi-engineer, dozens of PRs**, with the Python-specific flip (V.3) being the
smallest and last step. It is **not** reachable by continuing the Part IV
frontend sweep.

**Recommendation.** Parts I and III concluded these pins stay closed *because*
they are shared and risky; §16.1's go/stop reached the same verdict for Part IV.
Unless ESBMC commits to the **repo-wide "IREP2-native frontend→goto pipeline"
initiative** as a first-class program (Phases V.1k/V.2/V.4/V.5 as shared
infrastructure under their own umbrella issue), the defensible target remains
Part IV's: **IREP2-internal where the resolved-type/CF invariants allow, legacy
at the seam by design.** If the initiative *is* greenlit, **Phase V.1k
(resolve-then-build) is the mandatory first spike** — it is the keystone, its
proof obligation (RV2) is the highest-value question to settle, and every other
converter phase is blocked on it.
