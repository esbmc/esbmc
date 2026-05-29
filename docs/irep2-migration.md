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
