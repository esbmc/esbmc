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

1. **W** [DONE] `python_set.cpp:171` — `arr_type.size() - 1` (loop-length
   bound) → `sub2tc` via `gen_typecast_arithmetic`. Commit 23b46705be.
2. **W** [DONE] `python_set.cpp:234` — `idx < length_expr` loop condition →
   `build_less_than` over reconciled operands. Commit eff4bc08ae.
3. **W** [DONE] `list_comprehension.cpp:290` — `i < length` comprehension loop
   condition → `build_less_than` over reconciled operands. Commit 2dda75036c.
   (The doc's stale `python_list.cpp:4120/4533` line numbers; python-list is now
   split. Other list loop conditions — `list_access`, `list_construction:165`,
   `list_comprehension:494` — were already migrated by earlier V.3 sweeps.)
4. **W** [PARTIAL] `python_math.cpp` floor-div/modulo sign-correction trees —
   the four `value < 0` sign tests migrated via a shared `build_sign_test`
   helper. Commit 5642865dfa. REMAINING in these trees (deferred, entangled):
   the `mod(lhs,rhs)` node (nested width-hazard on lhs/rhs), the bool `xor`
   (must stay bool-xor, not bitxor — #4548 Bitwuzla crash), `and`, the modulo
   `if(cond, rhs, 0)` (mismatched branch widths), and the outer `+`/`-`.
4b. **W** [DONE] `list_mutation.cpp:935` — `idx < str_len` in
   build_extend_list_call. Commit fe23ac42b5. **With this, every clean
   loop-condition / sign-test width-hazard is drained.** A full census
   (`grep '("<"' etc. across src/python-frontend`) leaves only: the entangled
   `python_math` tree nodes (task 4 remaining) and one float site
   `numpy_call_expr.cpp:2503` (ieee_sub + rounding-mode — a different hazard).

5. **D** [PARTIAL] `builtins.cpp:369` isinstance is-None — DONE (commit
   9a7755247f). **Key finding: the F-P11 wall is milder than framed here.**
   `migrate(obj_expr)` resolves cleanly *even for a class-member operand*
   (`n.next`) — the converter resolves member/index sources inline at their
   construction (`build_member_expr_from_class`, `converter_expr.cpp:1160`), so
   by the time the operand reaches isinstance it is an already-resolved member,
   not a transient symbol_type. The existing list-pointer case
   (`builtins.cpp:391`) already migrated `obj_expr` — the precedent. Verified on
   member + variable operands, A/B byte-parity, full isinstance suite.
   REMAINING D sites, probe the same way (migrate + test the member-operand case
   before trusting): `builtins.cpp:447` isinstance tuple; BoolOp short-circuit
   (`converter_stmt.cpp`); is-none inequality (`converter_compare.cpp`);
   slice-bound arith (`python_list.cpp`).
6. **D** sites — status after probing:
   - [DONE] `converter_stmt.cpp` BoolOp `or` short-circuit `not` → not2tc
     (commit 02ffd18f01). Operand is the bool accumulator symbol — clean.
   - [RETAIN — not migratable] `converter_compare.cpp` is-none inequality and
     `builtins.cpp:447` isinstance-tuple OR-chain build **custom Python nodes**
     (`"isnone"`/`"isinstance"`) that `migrate_expr` cannot lower — they are
     resolved later by `simplify_python_builtins`, so they are a genuine retain
     boundary at construction, NOT a deferred-operand. **General lesson: a
     D-site is only migratable if it builds a standard expr (comparison/arith/
     bool) — sites that build custom `isinstance`/`isnone`/`isvoid` nodes stay
     legacy until simplify.**
   - [DONE] slice-bound arithmetic (`list_access.cpp`) — the size_add/size_sub/
     size_mul/size_div lambdas + char-array null-term add → IREP2 (commit
     60395caf8c). NOT a width-hazard: every operand is provably size_type
     (process_bound/to_size_expr/array_len/logical_len all yield size_type), so
     no reconciliation — verified by inspection (arith assert is release-inert;
     CI DebugOpt is the asserts-on gate). 56-test slice suite + A/B parity.
   - [TODO, probe] `converter_expr.cpp:1369`/`:1388` subscript-base derefs
     (deref, probe member-operand); `numpy_call_expr.cpp:1607` complex int→double
     (float — risky, rounding-mode).

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
