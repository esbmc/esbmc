# Architecture deepening backlog

Persistent memory for the `pm-deepen` routine. Each `## <slug>` is a candidate; `### Run` blocks under a slug (or under `## Run log`) are exit reports. Statuses: proposed | in-flight | landed | dropped | rejected. Never delete an entry — change its status.

## constant-kind-dispatch-seam

- **Status**: proposed
- **Score**: 22/25 (leverage 4, locality 4, blast radius 1, heat 5)
- **Files**: ~2 estimated
- **Modules**: `src/util/expr/expr_simplifier.cpp`, `unit/util/simplify2t.test.cpp`
- **Summary**: Collapse the four duplicated constant-kind ladders (`simplify_arith_2ops`, `simplify_logic_2ops`, `simplify_constant_relation`, `simplify_arith_1op`) into one `dispatch_binary_fold` seam driven by a per-kind trait, keeping each caller's extras caller-side.
- **First seen**: 2026-09-02
- **PR**: —

## string-method-dispatch-single-seam

- **Status**: proposed
- **Score**: 22/25 (leverage 4, locality 4, blast radius 1, heat 5)
- **Files**: ~3 estimated
- **Modules**: `src/python-frontend/string/string_handler.cpp`, `string/string_method_dispatch.h`, `string/string_method_handler.cpp`
- **Summary**: Collapse ten forwarding string-method dispatchers into one `method_name → handler` table with a single context struct and one keyword-unpacking guard.
- **First seen**: 2026-09-02
- **PR**: — (runner-up candidate this run; tied at 22/25, lost tie-break on recency + heavier end-to-end-only test gate — natural next firing)

## list-type-registry-single-seam

- **Status**: proposed
- **Score**: 22/25 (leverage 5, locality 5, blast radius 3, heat 4)
- **Files**: ~17 estimated
- **Modules**: `src/python-frontend/python-list/python_list.h`, `src/python-frontend/python-dict/*`, `converter/converter_stmt.cpp`, `function_call/expr.cpp`
- **Summary**: Hide the shared mutable `list_type_map` static behind a `list_type_registry` facade with a namespaced dict sub-API, removing the `friend class python_dict_handler` and the `$dict_value_types$` magic keys.
- **First seen**: 2026-09-02
- **PR**: — (highest architectural leverage; blast radius 3 loses the tie-break and is risky for one unattended PR — stageable)

## function-call-expr-god-object

- **Status**: proposed
- **Score**: 21/25 (leverage 5, locality 4, blast radius 4, heat 5)
- **Files**: large — must be sliced one family per PR
- **Modules**: `src/python-frontend/function_call/expr.h`, `expr.cpp` (7012 lines), `numpy/numpy_call_expr.h`
- **Summary**: Promote `get_dispatch_table` to a builtin registry with per-family deep handlers; make `numpy_call_expr` a registered handler rather than a subclass, retiring the `protected` hooks and the test-access `friend`.
- **First seen**: 2026-09-02
- **PR**: — (not one-PR work; schedule per-family slices as separate candidates)

## reassociate-chain-seam

- **Status**: proposed
- **Score**: 19/25 (leverage 4, locality 4, blast radius 1, heat 2)
- **Files**: ~3 estimated
- **Modules**: `src/util/expr/expr_reassociate.cpp`, `expr_reassociate.h`, caller `src/util/expr/expr_simplifier.cpp`
- **Summary**: Collapse the five near-identical `reassociate_*` bodies into one `reassociate(expr2tc&)` entry point that routes by kind, removing the caller's duplicated five-way dispatch and its reassociable-kind gate.
- **First seen**: 2026-09-02
- **PR**: — (cleanest refactor of the set, but `expr_reassociate.cpp` is cold — 1 commit/90d, last 2026-07-25 — so YAGNI down-weights heat)

## numpy-view-methods-out-of-python-list

- **Status**: proposed
- **Score**: 17/25 (leverage 3, locality 3, blast radius 2, heat 4)
- **Files**: ~4-6 estimated
- **Modules**: `src/python-frontend/python-list/python_list.h`, `python-list/list_access.cpp`, `numpy/numpy_call_expr.cpp`
- **Summary**: Move the ~14 NumPy ndarray-view construction methods out of `python_list` (where they are `public` only for `numpy_call_expr`) into a `numpy_view_builder`, shrinking the list interface.
- **First seen**: 2026-09-02
- **PR**: —

## irep2-adjust-mode-seam

- **Status**: dropped
- **Score**: not scored — hard-filtered
- **Files**: n/a
- **Modules**: `src/clang-c-frontend/clang_c_adjust_irep2.cpp`, `clang_c_adjust_irep2.h`
- **Summary**: Separate the `sole_adjuster` boolean mode / unify the legacy and irep2 adjusters behind a shared kind→handler seam.
- **First seen**: 2026-09-02
- **Reason**: Leverage 1 — fails the deletion test. `clang_c_adjust_irep2` is a transitional bridge the header states will replace and delete the legacy `clang_c_adjust`; deepening across the two adjusters spreads logic into a seam scheduled for demolition (complexity moves, not concentrates). The genuinely-shallow `statement_location` sub-item is only fixable in the central published `irep2.h` (out of scope). Machine judgement — reversible if the irep2 migration completes and the bridge stops being transitional.

## Run log

### Run 2026-09-02 — in progress

- **Outcome**: (updated at completion)
- **Stopped at**: —
- **Branch**: `sym/esbmc/routine/refactor-audit/01M1FK9R1B` (adopted — non-default, 0 commits ahead of origin/master, no upstream, unpublished on origin)
- **Committed**: report + backlog
- **Evidence**: first run — no prior `.architecture/`, `CONTEXT.md`, or `docs/adr/`
- **Next**: implement `constant-kind-dispatch-seam` test-first and open a PR
