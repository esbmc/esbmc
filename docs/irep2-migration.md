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
(`symbol.h:71-85`). The two setters behave asymmetrically:

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
`symbol.h:71-85`, `symbol.cpp:36-55,82-141`. A frontend *may* legally
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
