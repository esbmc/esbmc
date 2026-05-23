# V-track — Symbol-Value Source-of-Truth Decision

> **Status: complete.** V1 (#4737) closed the `migrate_expr_back`
> coverage gap for the 5 missing kinds. V2 (#4741) flipped
> `symbolt::value` to IREP2 storage with a lazy legacy cache, designed
> lazy from day one. End-state ABI recorded in
> `irep2-symbol-table-s6-plan.md`.

Plan for the value-side counterpart to Phase 5 of the IREP2 symbol-
table migration (esbmc/esbmc#4715). Phase 5 (#4732, #4735) flipped
`symbolt::type` to IREP2 storage and left `symbolt::value` exactly as
S4b shipped it: legacy `exprt` is the source of truth, the IREP2 form
is a lazy cache. This plan audits what would have to change to flip
the value side too, and recommends the path forward.

The plan is intentionally code-grounded: every count below comes from
a grep against the current `master`, listed as a one-liner so a
reviewer can reproduce.

## Status entering V-track

| Step | PR | Effect |
|------|----|--------|
| S5a — type-side flip | [#4735](https://github.com/esbmc/esbmc/pull/4735) | `type2tc type_` is the stored field; legacy `typet` is a lazy cache |
| S4b (still in place) | [#4730](https://github.com/esbmc/esbmc/pull/4730) | `value2_cache` / `value2_valid` on `symbolt`; `migrate_symbol_value` returns the cache |

`symbolt`'s value-side layout today (post-S5a):

```cpp
exprt           value;        // <-- source of truth
mutable expr2tc value2_cache; // <-- lazy IREP2 cache
mutable bool    value2_valid;
```

## The blocker, re-measured

The Phase 5 plan (#4732) presented `migrate_expr_back` as a switch
whose default arm aborts on every code-statement kind. After auditing
the actual coverage, **the blocker is much narrower** than that:

```
$ grep -E "^IREP2_EXPR\(" src/irep2/expr_kinds.inc | wc -l
109                            <-- total expr2t kinds
$ awk '/^exprt migrate_expr_back/{go=1} go && /^}$/{exit} go' \
    src/util/migrate.cpp | grep -oE "case expr2t::[a-z_]+_id" | sort -u | wc -l
104                            <-- already covered by migrate_expr_back
```

**5 kinds** are uncovered:

| Kind | Source | Encountered when… |
|---|---|---|
| `code_block2t` | function bodies, compound statements | every function symbol's `value` field |
| `code_cpp_catch2t` | C++ `catch` block | C++ programs only |
| `code_cpp_throw_decl2t` | C++ throw declaration | C++ programs only |
| `code_cpp_throw_decl_end2t` | end marker | C++ programs only |
| `pointer_capability2t` | CHERI-C pointer capability | CHERI-C builds only |

`code_block2t` is the only one that blocks a generic value-side flip.
The C++ throw-decl kinds and `pointer_capability` are
language-specific and trivially translatable (each is a small
operand vector or scalar).

## Caller audit — what reads `symbolt::value` today

```
$ grep -rE "\.get_value\(\)|->get_value\(\)" src/ --include="*.cpp" --include="*.h" | wc -l
195
```

195 call sites across `src/`, with this distribution:

| Tree | Callers | Notes |
|------|--------:|-------|
| `python-frontend/` | 118 | Constant-propagation and value-tracking on data symbols (initialisers, list contents, dict literals). Dominates the count. |
| `util/` | 23 | `c_link`, `fix_symbol`, formatting helpers. |
| `goto-programs/` | 19 | Includes 5 of the 8 body-reading sites (`goto_convert_functions.cpp`, `goto_main.cpp`). |
| `clang-c-frontend/` | 9 | Mostly initialisers. |
| `solidity-frontend/` | 9 | Same shape. |
| `clang-cpp-frontend/` | 5 | Same shape. |
| `goto-symex/` | 5 | Value reads on data symbols. |
| `esbmc/` | 1 | The function-call setup in `esbmc_parseoptions.cpp`. |

By **shape** of access:

| Pattern | Count | Implication for the flip |
|---------|------:|--------------------------|
| `to_code(...)` / `is_code()` / `get_statement()` (body) | **8** | The blocker. Needs `migrate_expr_back(code_block2t)` to derive a legacy `codet`. |
| `is_nil()` / `is_not_nil()` | 31 | Discriminator only; works with either form (the IREP2 side already has `is_nil_expr`). |
| `.type()` access on the value | 25 | Reads the **typet** of an exprt-shaped value (initialisers carry types). After the flip, becomes `get_value2()->type`. |
| Read-modify-set on local copy | ~30 | Already routed through `set_value` since S4a; the LHS / RHS exprts are short-lived legacy values that don't depend on storage. |
| Whole-value bind to `exprt` | many | The "bulk" of the 195. Most are short-lived legacy copies of initialisers / constants. |

The eight body-reader sites are the same set listed in the Phase 5
plan (#4732), with one site updated:

```
src/esbmc/esbmc_parseoptions.cpp:3014       to_code(fn->get_value())
src/goto-programs/goto_convert_functions.cpp:106   symbol.get_value().is_code()
src/goto-programs/goto_convert_functions.cpp:112   to_code(symbol.get_value())
src/goto-programs/goto_convert_functions.cpp:116   to_code(...).get_statement()
src/goto-programs/goto_convert_functions.cpp:422   collect_expr(s.get_value(), names)
src/goto-programs/goto_convert_functions.cpp:433   collect_expr(s.get_value(), list)
src/goto-programs/goto_main.cpp:31                  goto_convert(to_code(s->get_value()), ...)
src/python-frontend/python_list.cpp:3010            size_var->get_value().is_code()
src/python-frontend/converter/converter_funcall.cpp:54,56  is_code() + to_code()
```

Two of them (`python_list.cpp` and one branch in `converter_funcall.cpp`)
only use the body-typed value as a **discriminator** (`is_code()` to
gate "this symbol holds a function body, skip it"). For those, an
IREP2-side `is_code_type(symbol->get_value2()->type)` is a drop-in
replacement that does not require body back-migration.

## Three design options, re-scored

| Option | What changes | What it costs | What it unlocks |
|---|---|---|---|
| **A — Extend `migrate_expr_back` for the 5 missing kinds, then flip value storage** | Add 5 switch arms (each a small operand-vector inversion); flip storage mirror of S5a; one-time audit of 195 callers | ~50 lines in `migrate.cpp` + the S5a-mirror PR | Symmetric storage; opens the path to S6 (drop legacy `value` field) |
| **B — Keep value legacy permanently; remove the IREP2 value cache** | Drop `value2_cache` / `value2_valid` from `symbolt`; `migrate_symbol_value` recomputes per call (rare path — only one caller today) | Trivial (~20 lines deleted); marginal perf cost on contracts.cpp's one site | Acknowledges value-side staying legacy as the durable design |
| **C — Body carve-out** | `value` becomes IREP2-storage for non-bodies, legacy for bodies | Adds `is_code(type)` branching to every accessor; tag-discriminated storage | Half a step; rejected — same conclusion as the Phase 5 plan |

### Why Option A is now feasible

The Phase 5 plan rejected the symmetric-flip option on scope grounds:
*"adding `code_block2t` to `migrate_expr_back` means picking
back-direction representations for ~20 statement kinds."* The audit
shows that estimate was wrong — **104 of the 109 kinds are already
back-mappable**, and the 5 outliers are tightly bounded:

- `code_block2t` is a flat `std::vector<expr2tc>`. Back-mapping is
  build a legacy `code_blockt` and `move_to_operands` each
  back-migrated operand. ~10 lines.
- `code_cpp_catch2t` / `code_cpp_throw_decl2t` /
  `code_cpp_throw_decl_end2t` follow the forward-migration's
  symmetric shape — the forward direction picks the legacy `codet`
  with `statement("cpp-catch")` / etc., the back direction inverts
  it. ~30 lines combined.
- `pointer_capability2t` is a scalar operand on a typed value;
  inversion is mechanical. ~5 lines.

Total: ~50 lines of new switch arms, all structurally similar to the
existing 104.

### Why Option B is the lower-risk default

Option B accepts that **the value-side flip is not free** and that
the goto-binary on-disk format keeps a legacy `exprt` value field
regardless. The IREP2 value-cache on `symbolt` exists today only to
serve **one caller** — `contracts.cpp:499`'s
`migrate_symbol_value(contract_symbol, req)`. If that caller switched
to recomputing per call (it is on a slow path — contract
serialization, not the hot symex loop), the entire cache plus its
`value2_valid` book-keeping plus the swap / clear glue could be
deleted from `symbolt`. The legacy `exprt value;` field would remain
the source of truth permanently.

This is the **smallest possible delta** that ends the "what next?"
question for the value side. It is also reversible: if a future need
for IREP2-native value storage appears, the cache (or full flip) can
be re-introduced.

## Recommendation

**Pursue Option A in two slices, with B as a safety net.**

The decisive argument is the audit: the back-migration gap is 5
kinds, all tractable. The plan can be:

### V1 — Extend `migrate_expr_back` for the 5 missing kinds

Single PR against `src/util/migrate.cpp` plus
`unit/util/migrate.test.cpp` round-trip tests for each new kind.

Gates:
- `migrate_expr(migrate_expr_back(e)) == e` round-trip holds for each
  of the 5 kinds (extends the existing test pattern from #4722).
- `migratetest` stays green; `irep2test` stays green.
- No behaviour change anywhere else in the codebase — these arms are
  unreachable today except via the new tests.

Cost: ~50 lines of new code + ~30 lines of unit tests. Low risk,
because nothing in the pipeline actually calls these arms yet — they
only become reachable when V2 lands.

### V2 — Flip value storage (mirror of S5a)

After V1 is in master, repeat the S5a pattern for `value`:
- `expr2tc value_` becomes the stored field (source of truth).
- `mutable exprt legacy_value_cache_` derived lazily via
  `migrate_expr_back` on the first `get_value()` call.
- Legacy setters forward-migrate + populate the cache eagerly.
- `to_irep` / `from_irep` bridge via `get_value()` (no format change).
- `value2_cache` / `value2_valid` go away — `get_value2()` returns
  `value_` directly, mirror of how `get_type2()` works after S5a.

Pre-V2 regression test (mirror of #4733): add unit tests that pin
`get_value()` parity between the legacy and IREP2 representations on
function-body symbols and ordinary data symbols (including the
dual-role hazard with same-named globals).

### Fallback — Option B, if V1 surfaces unexpected complications

If extending `migrate_expr_back` for `code_block2t` turns out to have
hidden corners (e.g. the legacy `codet` carrying fields the IREP2
form does not preserve), retreat to Option B: drop the IREP2 value
cache, keep value-side legacy permanently. The audit gives no
indication this will be necessary, but it is the documented
escape hatch.

## Why not just leave it at S5a

S5a halved the legacy `irept` footprint on `symbolt` (the type
field). The value field is on the same scale. Completing the flip
brings the symbol-table migration to a clean end state — one where
`symbolt`'s authoritative storage is uniformly IREP2 — and unblocks
the eventual S6 step (drop both legacy fields entirely, decide on
the `const T&` vs `T` return ABI for the accessors).

If V1 turns out to be the ~50-line PR the audit predicts, the
remaining symbol-table migration is two more focused PRs from a
clean finish. That is worth doing.

## Validation strategy

Same shape as #4729 / #4730 / #4735:

1. **Cross-check.** Extend `migrate_symbol_value`'s NDEBUG assert to
   fire on every symbol value read once V2 lands. Currently it skips
   `is_code` values; after V1 lifts the blocker, the skip narrows to
   `is_nil` only.
2. **Differential goto-binary diff.** Run
   `scripts/irep2-migration/capture_goto_baseline.sh` on master,
   `diff_goto_baseline.sh` on V2's branch. Zero diff per slice.
3. **Unit tests.** `migratetest`, `irep2test`, `symboltest` all stay
   green. V1 adds 5 new round-trip cases.
4. **Sanity verdicts.** Same C assert / C OOB / W/W race set.
5. **Dual-solver.** Bitwuzla + Z3 agreement on the smoke set.
6. **Python sanity.** `scripts/check_python_tests.sh` — important
   here because `python-frontend/` owns 60 % of value-side callers.
7. **Read-from-old-binary.** Goto-binary written by master must
   load under V2.

## Risk register

| Risk | Mitigation |
|---|---|
| The 5 new back-migration arms produce a legacy `codet` whose shape differs subtly from the forward-migrated original | Round-trip property `migrate_expr(migrate_expr_back(e)) == e` is the explicit gate in V1's unit tests. |
| `code_block2t` operands include kinds that themselves need new back-arms | Audit recursive: `migrate_expr_back` already covers every operand kind a body can contain (all 109 kinds reachable inside a code block are in the 104 covered set). The V1 PR will state this explicitly with a one-liner check. |
| Python-frontend value-side caller volume (118 sites) hides a regression | `scripts/check_python_tests.sh` plus the python regression label catches behaviour change; the V2 cross-check fires on every read. |
| `from_irep` on old binaries that wrote a default/empty exprt value | Handled the same way S5a handles a default `typet`: nil legacy ⇒ nil IREP2 source. The pre-V2 state is preserved. |
| Body-reader sites that take `const codet &` from `get_value()` see a value backed by a lazily-derived cache whose lifetime is the symbol's | Same lifetime guarantee S5a uses for the legacy `typet` cache; no caller binds across symbol-destructor boundaries today. |

## Rollback

- V1 is dead code until V2 lands — independently revertable.
- V2 is a single-file flip of `symbolt`'s value storage + the
  matching `migrate.cpp` adjustment. Revert is a single PR revert.
- No on-disk format change at any step.
- The legacy goto-binary reader remains permanently.

## Position relative to Phase 5

Phase 5 (Option B in #4732) was the type-only flip. This V-track
plan adopts Option A for the value side, justified by the audit's
finding that the back-migration gap is 5 kinds rather than the
~20 the Phase 5 plan estimated. **The two plans are independent**:
V-track does not change the type side; Phase 5's S5b decision
(whether to drop the legacy `typet` cache or keep it permanently)
is unaffected and stays deferred.

## Recommended order

1. **V1** — extend `migrate_expr_back` for the 5 missing kinds, with
   unit-test round-trip gates. Dead code in the pipeline. Mergeable
   on its own.
2. **Pre-V2** — value-side discriminator-parity unit tests, mirror
   of #4733.
3. **V2** — flip value storage; mirror of S5a.
4. **S6 (separate plan)** — drop both legacy `typet` and `exprt`
   fields, finalise the accessor ABI. Only after V2 has settled.

This document is the recommendation; V1 is the next code PR.
