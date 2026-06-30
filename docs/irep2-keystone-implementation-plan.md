# V.1k keystone — implementation plan (feature branch)

Working plan for the IREP2-native resolution keystone tracked in
`docs/irep2-migration.md` (V.1k / V.3 residual). This file lives on
`feat/python-irep2-adjust-keystone` and is the execution roadmap; the
canonical record stays in `irep2-migration.md`. Delete on merge.

## What the spike established (see irep2-migration.md, "Spike result 2026-06-30")

The residual that blocks the converter from building IREP2 directly is two
body-wide classes, **neither a pure tag-following problem**:

1. **Deferred-resolution operands** — enclosing sites (`isinstance`/is-none
   `builtins.cpp:369`, `BoolOp`) build over an operand that is itself a
   member/index subexpression whose source resolution is deferred to the
   body-wide `clang_cpp_adjust` pass.
2. **Width-hazard arithmetic/relational** — `<`/`-`/`!=` over concrete but
   mismatched widths that the legacy `exprt` tolerates (because
   `clang_cpp_adjust::adjust_expr_rel`/`adjust_expr_binary_arithmetic` insert
   the reconciling typecast via `gen_typecast_arithmetic`) and
   `lessthan2t`/`sub2t` reject at construction.

## Key implementation insight (new this iteration)

IREP2's construction-time asserts mean **most of `clang_cpp_adjust`'s
completion must happen at converter construction, not in a post-pass**:

- A `member2t`/`index2t` cannot be built over a pointer/array source (the
  assert forbids it), so `adjust_member`'s base-wrapping
  (`clang_c_adjust_expr.cpp:296`-`312`) is structurally N/A — the converter
  already derefs/indexes before constructing (`build_member_expr_from_class`,
  `converter_expr.cpp:1160`).
- Arith/relational nodes cannot be built width-mismatched, so width
  reconciliation cannot be deferred to `python_adjust` — it must run inline at
  construction.

Therefore the only completion deferrable to the `python_adjust` pass is
**symbol_type source following** (via the relaxed assert), which B.1 already
does. The keystone's remaining work is **converter-side inline resolution**:

- **(W) Width-hazard sites:** before building the IREP2 node, reconcile
  operand widths with the *exact* `gen_typecast_arithmetic(ns, op0, op1)`
  (`src/clang-c-frontend/typecast.h`) that `clang_cpp_adjust` would apply — a
  guaranteed byte-identical, idempotent transform (not a blind cast, which the
  plan correctly flags as unsound). Then `migrate` + build `sub2tc`/`lessthan2tc`
  + `migrate_expr_back`.
- **(D) Deferred-operand sites:** resolve the operand's member/index
  subexpression types (`ns.follow`) before migrating it, so the enclosing IREP2
  node builds over a resolved operand.

## Ordered site list (W = width-hazard, D = deferred-operand)

Each lands as its own commit, gated on dual-solver (Bitwuzla + Z3) verdict +
matched-text parity over the affected regression suite, asserts build.

1. **W** `python_set.cpp:171` — `arr_type.size() - 1` (the loop-length bound).
   Smallest self-contained site; the loop `<` at `:234` stays legacy for now.
2. **W** `python_set.cpp:234` — `idx < length_expr` loop condition.
3. **W** `python_list.cpp:4120`, `:4533` — analogous list guards.
4. **W** `python_math.cpp:674`-`753` — floor-div/modulo sign-correction trees
   (relational + arith leaves; larger, migrate the whole correction subtree).
5. **D** `builtins.cpp:369`/`447` — `isinstance` NoneType/tuple.
6. **D** `converter_stmt.cpp` BoolOp short-circuit; `converter_compare.cpp`
   is-none inequality; `python_list.cpp:1444`-`1683` slice-bound arithmetic;
   `numpy_call_expr.cpp:1607` complex int→double; `converter_expr.cpp:1369`/
   `:1388` subscript-base derefs.

## Verification protocol per commit

1. Rebuild `esbmc` (one TU + link for converter-only edits).
2. Run the affected `regression/python` subset under **both** Bitwuzla and Z3;
   require identical verdict + matched counterexample text vs. pre-patch.
3. `scripts/check_python_tests.sh <subset>` CPython sanity.
4. Only commit on parity; revert otherwise (incremental-patch-testing rule).

## Done

When W + D sites build IREP2 in-converter with zero `irep2_expr.h` aborts and
the whole `regression/python` corpus holds dual-solver parity, the converter's
expression surface is IREP2-native and §V.1 acceptance bars #1/#2 fall for it.
`python_adjust` remains the thin symbol-following pass (B.1); the legacy
`clang_cpp_adjust` round-trip is then removable from the Python path.
