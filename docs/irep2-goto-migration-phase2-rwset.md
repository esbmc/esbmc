---
title: "Phase 2 work breakdown — rw_set / data-race migration to IREP2"
status: draft
date: 2026-05-22
parent: docs/irep2-goto-migration-plan.md
tracking-issue: esbmc/esbmc#4715
---

# Phase 2 expanded: migrate `rw_set` (data-race detection) to IREP2

Phase 2 from the roadmap, broken into four PR-sized, independently reviewable
and revertible units. Each PR has an explicit scope, the exact edits, the
behaviour-preservation argument, and its own gate.

**Why this phase is the template.** `rw_set` is the most self-contained legacy
island (B5) — but it still has a B2-coupled tail (symbol *creation* in
`w_guardst`). So it demonstrates the general pattern for the whole migration:
*migrate the computation cleanly, then stop at the symbol-table boundary and
defer that tail to Phase 4.* Do not let PR 2.4 below tempt you into starting B2
early.

## Current data flow (ground truth)

```
add_race_assertions (per instruction)
  instruction.code / .guard  : expr2tc                      ← already IREP2
    └─ migrate_expr_back  →  exprt  ────────┐  (round-trip #1, add_race:130-136)
                                            ▼
  rw_sett::compute(const exprt&)            ← LEGACY island (rw_set.cpp:8)
    └─ read_write_rec(... guard2tc ...)     ← guard already IREP2 internally
         └─ entryt{ exprt guard;            ← stored back as exprt via
                    exprt original_expr }     migrate_expr_back (rw_set.cpp:125)
                                            │
  w_guardst::get_guard_symbol_expr(exprt&)  ← builds exprt("races_check") …
    └─ migrate_expr  →  expr2tc  ───────────┘  (round-trips #2/#3, add_race:174,191,239)
                                               then symex consumes races_check2t
```

`races_check2t` is already a first-class IREP2 node (`src/irep2/irep2_expr.h:180,984`)
and symex already consumes it as `expr2tc` (`src/goto-symex/builtin_functions/misc.cpp:24-30`).
So every `migrate_expr`/`migrate_expr_back` above is a pure round-trip — except
the symbol-table writes in `w_guardst` (`migrate_type_back` at
`add_race_assertions.cpp:35,85`), which are forced by B2.

---

## PR 2.1 — Migrate `rw_sett` internals + `entryt` storage to IREP2

**Scope:** `src/goto-programs/rw_set.h`, `rw_set.cpp`, and the minimal
consumption changes in `add_race_assertions.cpp` needed to keep it compiling.
Leave `w_guardst`'s legacy expr building intact (adapt at its boundary).

**Edits:**
1. `rw_set.h:17-18` — change `entryt`:
   - `exprt guard;` → `guard2tc guard;` (it is *already* produced from a
     `guard2tc` at `rw_set.cpp:125`; just store it directly).
   - `exprt original_expr;` → `expr2tc original_expr;`
   - `entryt()` initialiser `guard(true_exprt())` → default `guard2tc()`.
   - `get_guard()` returns `const guard2tc &`.
2. `rw_set.h:46,56,62-71,77,79-86` — retype signatures from `const exprt &` to
   `const expr2tc &`: `compute`, the convenience ctor, `read_rec` (both),
   `assign`, `read_write_rec`. The `guard2tc guard` parameter already exists.
3. `rw_set.cpp:8-58` `compute` — replace the stringy
   `expr.is_code()` / `code.get_statement()` dispatch with an IREP2 node switch:
   - `"assign"` → `is_code_assign2t` → `assign(asn.target, asn.source)`
   - `"printf"` → `code_printf2t` / `code_expression2t` operand walk
   - `"return"` → `code_return2t` → `read_rec(ret.operand)`
   - `"function_call"` → `code_function_call2t` (`.function`, `.operands`,
     symbol-body check via `ns.lookup`). Preserve the
     `symbol->value.is_nil() || name == "__VERIFIER_assert"` guard exactly
     (`rw_set.cpp:44`).
   - the `is_goto()/is_assert()/is_assume()` branch (`:55-57`) is unchanged.
4. `rw_set.cpp:66-183` `read_write_rec` — replace the `is_symbol/is_member/
   is_index/is_dereference/is_address_of/"if"` chain with
   `symbol2t/member2t/index2t/dereference2t/address_of2t/if2t`. The internal
   guard handling at `:165-176` already uses `guard2tc`; **delete** the
   `migrate_expr` calls at `:166-173` (build the if-condition guard directly in
   IREP2 — `true_guard.add(if_expr.cond)`, `false_guard.add(not2tc(if_expr.cond))`).
   - **Preserve the symbol filters verbatim** (`:87-115`): the Python-global
     `mode == "Python" && !file_local` rule, the `__ESBMC_alloc`/`stdin`/… name
     list, CUDA `indexOfThread`/`indexOfBlock`, and the `#atomic` C11 filter.
     These read `symbolt` fields (`mode`, `file_local`, `static_lifetime`,
     `is_thread_local`, `type.get_bool("#atomic")`) which are unaffected by this
     PR (B2 is untouched).
   - **delete** `entry.guard = migrate_expr_back(guard.as_expr())` → `entry.guard = guard;`
5. `add_race_assertions.cpp:130-136` — drop the `migrate_expr_back` of
   `instruction.guard`/`instruction.code`; pass them directly:
   `rw_sett rw_set(ns, i_it, instruction.is_goto()||instruction.is_assert() ? instruction.guard : instruction.code);`
6. `w_guardst` boundary adaptation (kept legacy in this PR): where it reads
   `entry.original_expr` (now `expr2tc`) it back-migrates locally —
   `get_guard_symbol_expr(migrate_expr_back(entry.original_expr))` — and
   `entry.get_guard()` (now `guard2tc`) is back-migrated at
   `add_race_assertions.cpp:189`. These two local back-migrations are removed in
   PR 2.2.

**Behaviour-preservation argument:** every transformation is a structural
1:1 re-expression; the guard was already `guard2tc` internally, so dropping the
back-migration at `:125` removes a lossy round-trip, not a computation. The
symbol filters and the `original_expr` nil-fallback logic (`:126,132,146,152`)
are preserved exactly (`is_nil()` → `is_nil(expr2tc)`).

**Gate:** zero goto-binary diff on the concurrency corpus
(`ctest -L concurrency`, `--data-races-check` tests), dual-solver Bitwuzla+Z3
agreement, ASan/UBSan clean (lifetime risk — `expr2tc` COW now in play).

---

## PR 2.2 — Migrate `w_guardst` expression *building* to IREP2

**Scope:** `add_race_assertions.cpp:44-64` (`get_guard_symbol_expr`,
`get_w_guard_expr`, `get_assertion`) and the three `migrate_expr` call sites that
consume their results (`:174,191,239`). Symbol *creation* stays legacy (PR 2.4).

**Edits:**
1. `get_guard_symbol_expr` — build IREP2 directly:
   `races_check2tc(address_of2tc(orig->type, orig))` instead of
   `exprt("races_check")` + `move_to_operands`. Return `expr2tc`.
   (`races_check2t` exists: `irep2_expr.h:180,984`; symex consumes it at
   `builtin_functions/misc.cpp:24`.)
2. `get_assertion` returns `not2tc(get_guard_symbol_expr(...))`.
3. `add_race_assertions.cpp:173-175` — `t->make_assertion(w_guards.get_assertion(...))`
   directly (drop `migrate_expr`).
4. `:188-191` and `:237-239` — build `code_assign2tc(w_guards.get_w_guard_expr(...),
   entry.get_guard())` / `code_assign2tc(..., gen_false_expr())` directly (drop
   the legacy `code_assignt` + `migrate_expr`). Note `entry.get_guard()` is now
   `guard2tc` (PR 2.1) → use `.as_expr()`.
5. Remove the two local back-migrations introduced in PR 2.1 step 6.

**Behaviour-preservation argument:** `migrate_expr(exprt("races_check", bool, addr))`
produces exactly `races_check2tc(addr)` (`migrate.cpp:1688-1692`); constructing
it directly is byte-identical post-migration. The assertion `gen_not` → `not2tc`
and `code_assignt` → `code_assign2tc` are the same nodes the migrator emits.

**Gate:** zero goto-binary diff (this is the strongest check here — the inserted
instruction stream must be identical); dual-solver agreement on the concurrency
corpus.

---

## PR 2.3 — Tests: differential + targeted regression

**Scope:** `regression/esbmc/` (or `regression/concurrency/`) and `unit/`.

**Edits:**
1. Add a focused regression pair under `regression/esbmc/github_4715_rwset/`
   (one CORE pass, one `*_fail` race-positive) exercising: array-index race
   (`original_expr.is_index()` path), member race, `if`-guarded race
   (the `if2t` guard split), and a `__VERIFIER_assert` function-call read.
   These pin the branches PR 2.1/2.2 rewrote.
2. Unit test in `unit/goto-programs/` constructing a small `goto_programt`,
   running `rw_sett::compute` on hand-built `expr2tc`, and asserting the
   `entries` map (objects, r/w flags, deref) matches a golden — no migration in
   the test, proving the IREP2 path standalone.
3. Wire the **differential goto-binary harness** (Phase 0) target for the
   concurrency corpus into CI as a non-blocking report initially.

**Gate:** new tests pass on the PR 2.1+2.2 build and *fail* if reverted to
master (proving they discriminate).

---

## PR 2.4 — DEFERRED to Phase 4 (do not implement now)

The symbol creation in `w_guardst::get_guard_symbol` (`add_race_assertions.cpp:31-41`)
and `add_initialization` (`:82-88`) writes `new_symbol.type = migrate_type_back(...)`
and `new_symbol.value.make_false()` into the `contextt`. This is forced by B2
(`symbolt` stores `typet`/`exprt`). **Leave it legacy.** When Phase 4 migrates
the symbol table, these two `migrate_type_back` calls and the `.make_false()`
become `symbolt::type2 = array_type2tc(...)` / `value2 = gen_false_expr()` and
this PR is folded into the Phase-4 writer switch (step 4c). Tracking note only;
no code in Phase 2.

---

## Phase 2 exit criteria (all required)

- `rw_set.{h,cpp}` contain **zero** `exprt`/`typet`/`migrate_*` (except the
  documented symbol-filter `symbolt` field reads, which are not expressions).
- `add_race_assertions.cpp` contains `migrate_*` calls **only** in the
  symbol-creation tail (PR 2.4 scope) — the per-instruction hot path is
  migrate-free.
- Concurrency corpus: **zero goto-binary diff** vs. the Phase-0 baseline.
- Dual-solver (Bitwuzla + Z3) verdict **and** counterexample-digest agreement.
- ASan/UBSan build of the concurrency regression subset is clean.
- `migrate_*` census (Phase 0 scoreboard) strictly decreases in
  `src/goto-programs/{rw_set.cpp,add_race_assertions.cpp}` and increases nowhere.

## Suggested PR order & rollback

```
PR 2.1  rw_set internals + entryt        ← revertible alone (w_guardst still legacy)
PR 2.2  w_guardst expr building          ← depends on 2.1; revert restores 2.1 state
PR 2.3  tests                            ← can merge before or after 2.2
PR 2.4  (none — folded into Phase 4)
```

Each PR is green-before-merge (no broken commits, no squashing). If PR 2.2's
goto-binary diff is non-zero and the cause is not an obvious justified
improvement, revert 2.2 only — 2.1 already removed round-trip #1 and stands on
its own.
