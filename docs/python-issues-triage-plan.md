# Python issues — triage & improvement plan

This document is a snapshot triage of the open `python`-labelled issues in the
ESBMC tracker, with a per-issue disposition (fix shipped / already resolved /
intentional behavior / design-level / multi-blocker) and a prioritised plan for
the remaining work. It exists so the next contributor does not have to
re-reproduce ~60 KNOWNBUG tests to know where the real, *sound* wins are.

- **Snapshot date:** 2026-05-31 (ESBMC 8.3.0, `master` HEAD).
- **Method:** pulled all 40 open `python` issues + recently-closed ones,
  reproduced every currently-failing Python KNOWNBUG test, classified by failure
  mode, and fixed the issues that admit a sound, contained fix.
- **Guiding constraint:** ESBMC is a formal-verification tool. An unsound or
  heuristic "fix" is worse than an honest KNOWNBUG, so forcing fixes on
  multi-blocker or deep-modeling issues was deliberately avoided.
- **How to refresh:** re-run
  `grep -lE '^KNOWNBUG' regression/{python,python-intensive,quixbugs,humaneval}/*/test.desc`
  and `gh issue list --repo esbmc/esbmc --label python --state open`, then
  re-reproduce. The cluster structure below is the stable part; exact counts
  drift as fixes land.

---

## 1. Fixes shipped

| Issue | PR | Title | Status |
|---|---|---|---|
| [#4984](https://github.com/esbmc/esbmc/issues/4984) | [#4993](https://github.com/esbmc/esbmc/pull/4993) | `[python]` carry referenced module globals through named imports | merged |
| [#4807](https://github.com/esbmc/esbmc/issues/4807) | [#4996](https://github.com/esbmc/esbmc/pull/4996) | `[python]` scope return-type inference to the enclosing function | merged — flips `humaneval_127` (C1) to `CORE` |
| [#4807](https://github.com/esbmc/esbmc/issues/4807) | _branch `feat/list-filter-rewrite`_ | `[python]` desugar `list(filter(pred, seq))` to a comprehension (plan §6 item 4) | validated, pending PR |

### #4984 — `from mod import C` drops module globals used by C's methods
- **Root cause:** `_filter_nodes_for_import` in
  `src/python-frontend/parser/json_emitter.py` pruned the imported module's AST
  to the imported name before conversion, keeping referenced sibling
  **classes/functions** but dropping module **globals** (`AnnAssign`/`Assign`)
  unless explicitly imported. `get_referenced_names` also never collected bare
  `Load` reads like `TAG`. So `from modstub import C` emitted `[ClassDef C]` with
  `TAG: int = 1` dropped → `Variable 'TAG' is not defined`.
- **Fix:** retain the **transitive closure** of top-level defs/globals referenced
  by the imported symbols; broaden `get_referenced_names` to all `Load`-context
  reads. The closure only ever **adds** nodes (never drops) — the safe direction
  for a verifier — and matches CPython LEGB (a method resolves a module global
  against its defining module at call time). Also fixes latent gaps: transitive
  helper deps and async-function retention.
- **Validation:** 3 new regression tests — `github_4984` (direct global →
  SUCCESSFUL), `github_4984_fail` (mismatched value → FAILED, guards against a
  dropped/nondet global), `github_4984_transitive` (`C`→helper→global, exercises
  the closure). All 94 multi-file Python import regression tests pass (the full
  at-risk surface; the change is inert for single-file/no-import programs).
  CPython sanity (`check_python_tests.sh`) ✓, pylint 10/10 ✓, code-review clean.

### `list(filter(...))` — eager-filter materialisation (plan §6 item 4)
- **Root cause:** the preprocessor (`expression_rewrite_mixin.py`) already
  desugars `list(map(f, xs))` to a comprehension and handles the
  `for x in filter(...)` statement form (`loop_mixin._transform_filter_for`),
  but `list(filter(pred, xs))` had no rewrite — it reached the converter as a
  call to the unmodelled `filter` builtin and verification stopped with
  `Unsupported function 'filter' is reached`.
- **Fix:** add a `list(filter(pred, seq))` → `[x for x in seq if pred(x)]`
  rewrite next to the `map` rewrite, the exact CPython desugaring. Three handled
  predicate forms: a one-arg `lambda` (its param becomes the comprehension
  target, its body the `if` guard), a named callable (`pred(t)` guard over a
  fresh `ESBMC_filter_elt_N` target), and `filter(None, seq)` (the element's own
  truthiness is the guard). Any other shape (wrong arity, keywords) falls through
  to `generic_visit`, preserving the honest `Unsupported` diagnostic — the safe
  direction for a verifier. The rewrite only ever **adds** an `if`-guarded
  comprehension; it never fabricates elements (the negative test pins this).
- **Validation:** 2 new regression tests — `list_filter` (lambda + named-callable
  + `filter(None,…)`, exact element/length assertions → SUCCESSFUL) and
  `list_filter_fail` (over-long expected length → FAILED, guards against a
  nondet/fabricated list). Existing `filter_loop{,_fail}`, `github_4294_filter`,
  and the map/listcomp suite still pass. CPython sanity
  (`check_python_tests.sh list_filter`) ✓, pylint clean (no new categories;
  E-level 10/10), code-review clean.
- **Benchmark flip:** re-tags `humaneval_136` (`largest_smallest_integers`, two
  `list(filter(lambda…))` calls) from `KNOWNBUG` to `CORE` — now
  `VERIFICATION SUCCESSFUL`, verified non-vacuous (wrong expected tuple → FAILED)
  and CPython-consistent. Does **not** flip humaneval 108/145/151 — those also
  need `str(int)` in a comprehension and `sorted(key=)` (see §5 C2); 108 is the
  only remaining KNOWNBUG that uses `filter`.
- **Note — `reversed(list)` is no longer a gap:** the §5 quixbugs note that
  `reversed()` on a list is unmodelled is **stale** for the `for x in reversed(seq)`
  form, which now lowers correctly (`loop_mixin._transform_reversed_for`,
  verified non-vacuously). `next_permutation` remains blocked by slice-assignment
  + list `==`, not by `reversed`.

---

## 2. Already resolved — recommend closing (no work needed)

These open issues' tests already pass / were already re-tagged on `master`:

| Issue | Evidence |
|---|---|
| #4774 `mixed-types` | test.desc is now `CORE`, runs `--incremental-bmc`, → **VERIFICATION SUCCESSFUL** locally. No longer KNOWNBUG. |
| #4770 `cast` (`chr(int(float))`) | no longer in the KNOWNBUG set on `master` (closed-issue list shows #4770 COMPLETED 2026-05-31). |
| #4777 `python-intensive/problem1_fail` | re-tagged `FUTURE` (acknowledged, excluded from default run). |

---

## 3. Intentional conservative behavior — do NOT "fix" without a redesign

The **github_4117** trio is documented *in the test sources themselves* as a
deliberate soundness-preserving limitation of the lexical usage-site type
scanner. Failure mode: `ERROR: Cannot resolve nested attribute: <field>`.

| Issue | Test | Why intentional |
|---|---|---|
| #4771 | `github_4117_attr_conflict` | an attribute assigned values of two classes → scanner refuses to commit (returns `any_type()`) rather than unsoundly pick one. |
| #4772 | `github_4117_attr_shadow` | a variable reassigned to a different class → scanner drops the class mapping rather than mis-attribute later writes. |
| #4773 | `github_4117_function_internal` | instance-var writes inside function bodies aren't tracked; needs *scope-aware* tracking. The only one without a conflict, but still requires flow-sensitive analysis to fix soundly. |

The same `Cannot resolve nested attribute` signature blocks **quixbugs/detect_cycle**
(`node.successor.successor`) and is the dominant frontend blocker for the
linked-list/graph quixbugs. **Required design change:** flow-sensitive
(per-program-point) class tracking instead of the current lexical scan, plus a
union-type or diagnostic model for genuine conflicts. High risk; out of scope
for a safe point fix.

---

## 4. Design-level issues — blockers documented, not point-fixable

| Issue | Summary | Blocker / required design change |
|---|---|---|
| #4642 | arbitrary-precision (bignum) int | needs IR-level wide-int support; #4653 is the OM-rework prerequisite. |
| #4653 | OM rework to unblock `--ir-gated` bignum | depends on #4642 IR work. |
| #4579 | model CPython GIL-aware thread scheduling | concurrency-semantics design effort. |
| #4584 | insert interleaving points at module-global accesses in Thread bodies | symex scheduling change; couples with #4579. |
| #4566 | `concurrency_fail`: needs `threading.Thread` + `queue.Queue` | blocked on the threading/Queue model (#4579/#4584). |
| #3067 | refactor class handling into a dedicated class | pure refactor, no behavior change. |
| #2848 | type inference for `typing.Any` at symex level | inference design. |
| #3541 | math dataset/benchmark suite | infra, not a bug. |

---

## 5. Benchmark KNOWNBUGs — full failure-mode classification

All reproduced on `master` HEAD. None is a clean single-root-cause fix that
soundly flips a verdict; each is multi-blocker, scalability-bound, an internal
assertion/segfault needing care, or the intentional limitation above.

### quixbugs (#4778 umbrella, 25 still failing)

- **Cluster 1 — `Cannot unpack <X>` (6, frontend conversion crash):**
  `minimum_spanning_tree(_fail)` (`signedbv`), `powerset(_fail)` (`pointer`),
  `shortest_path_length(_fail)` (`empty`). Tuple-unpack target (`u, v = edge`)
  where the RHS isn't inferred as a tuple/array. *Also* blocked by `heappop`
  (undefined → assert(false)) and rich dict/set semantics → fixing unpack alone
  won't flip them.
- **Cluster 2 — TIMEOUT/scalability (5):** `flatten_fail`, `knapsack(_fail)`,
  `next_permutation_fail`, `shortest_path_lengths`. Deep recursion / list-OM
  loops / SMT explosion (5418 VCCs). Not a frontend bug.
- **Cluster 3 — segfault in `__memcpy_impl` (string.c:278) during unwind (2):**
  `topological_ordering(_fail)`. Backend robustness bug; deep.
- **Cluster 4 — `NameError: name 'Queue' is not defined` (2):**
  `breadth_first_search(_fail)`. `queue.Queue`/FIFO unmodelled.
- **Singletons:** `detect_cycle` (`Cannot resolve nested attribute` — §3);
  `depth_first_search` (internal assert `is_array_type` in `object_size.cpp:179`);
  `reverse_linked_list` (bitwuzla `mk_eq` width-mismatch assert);
  `shortest_paths(_fail)` (`Only simple targets are supported in DictComp` —
  tuple target in a dict comprehension; also `float('inf')`+dict semantics);
  `next_permutation` (`reversed()` on a list unmodelled — only `reversed(range)`
  is lowered — plus slice-assignment + list `==`);
  `wrap` (unwinding assertion: `--unwind 200` too low for the string loop);
  `bitcount_fail` (k-induction UNKNOWN);
  `rpn_eval`/`rpn_eval_fail` — **`rpn_eval_fail/main.py` is invalid Python**
  (calls `rpn_eval(3.0, 5.0, '+', …)` — 5 args to a 1-arg function; CPython
  raises `TypeError`). The intended call passes a list. Both also use
  `--unwind 1`, too low for the token loop. Needs a test-config fix, not an
  ESBMC fix; murky verdict, documented for the maintainer.

### humaneval (#4807 umbrella, 30 still failing)

- **C1 — `assertion 0` on computed string/list `== literal` (4):** 62, 103, 119,
  125. (`127` fixed by #4807 / PR #4996 — return-type inference is now scoped to
  the enclosing function; it is `CORE`/SUCCESSFUL on `master`.) Likely a
  string/list-equality modeling issue (soundness-relevant),
  but intertwined with string concat, comprehensions, and per-function logic.
  Deep solver/model change — not safe to rush.
- **C2 — `TypeError: str() expects a string argument` (3):** 108, 145, 151.
  `str(int)` rejected by the str() model. Generally-useful gap, but each test
  also hits `filter`/`sorted(key=)`/comprehensions → won't flip on its own.
- **C3 — string-OM unwinding assertions (3):** 1, 1-1, 67. `--unwind` too low for
  `strlen`/`__python_strnlen_bounded`. Bound/loop-handling.
- **C4 — `Unsupported reassignment from dict to list` (2):** 93, 126.
- **C5 — `tuple(...) == tuple(...)` → `NONDET(_Bool)` (2):** 33, 37. Tuple-equality
  nondet (soundness), but tests also use slice-assignment, `zip`, `extend`,
  extended slices.
- **C6 — scalar return mismatches (≈4, distinct causes):** 59, 78, 95, 137 (+86).
- **Singletons:** 123 (sorted mixed types), ~~136 (`filter()`)~~ **fixed** —
  `list(filter(lambda…))` now lowers to a comprehension (§1), re-tagged `CORE`;
  148 (tuple slice
  non-const lower), 158 (list-OM empty type), 162 (`hashlib` unmodelled),
  2-1 (numpy `fmod`), 91 (`split` maxsplit), and three internal C++ asserts —
  29 (`to_array_type`), 39 (`get_significand_width`), 90 (`member2t`).

### regression/python (#4769 umbrella, 6 still failing)
`github_4117_attr_conflict/_shadow/_function_internal` (§3 intentional);
`rover` (`Variable 'Twist' not defined` — named-import-into-function scope +
`Dict[str,T]` + nested attr writes; multi-feature); `shedskin`
(`'int' object is not subscriptable` on `d.items()` + broad list/dict/str/set
smoke test); `concurrency_fail` (#4566, design).

---

## 6. Prioritised plan for the remaining work

Each item below is its own focused, carefully-validated effort — not a rush.
Ordered by leverage × soundness-confidence.

1. **Flow-sensitive class tracking.** Replace the lexical usage-site scanner with
   per-program-point class tracking (+ union-type / diagnostic for genuine
   conflicts). Unblocks #4771/#4772/#4773 + `detect_cycle` + the linked-list /
   graph quixbugs. Largest leverage, but a real redesign; must preserve the
   current conservative soundness on true conflicts.
2. **String/list-equality modeling (humaneval C1).** Soundness-relevant: a
   computed string/list compared to a literal currently collapses to false.
   Needs solver-level validation against CPython semantics before trusting any
   verdict change.
3. **Tuple-unpack-from-iteration type inference (quixbugs Cluster 1).** Infer the
   element type of `for u, v in <dict/sequence of tuples>` and `u, v = edge` so
   the unpack resolves to a tuple/array.
4. **Generally-useful builtins, each with its own regression pair.** Status:
   - `filter` — **done.** `for x in filter(...)` (`filter_loop`) and
     `list(filter(...))` (`list_filter`, this plan) both lower to comprehensions.
   - `reversed(list)` — **done** for the `for x in reversed(seq)` form
     (`_transform_reversed_for`); `list(reversed(seq))` not yet exercised.
   - `str(int)` — **partial.** `str(<int literal/var>)` works; the open gap is
     `str(n)` inside a comprehension after a tuple-unpack reassignment
     (`n, neg = -1*n, -1`), which currently aborts in the converter with an
     internal `assert_arith_2ops_consistency` (irep2_expr.cpp) rather than the
     `TypeError: str() expects a string argument` seen for the whole-function
     conversion. Entangled with tuple-unpack type tracking; not a one-line fix.
   - `queue.Queue` / FIFO — **not started** (blocks BFS quixbugs + #4566).

**Do NOT** machine-fix the scalability/timeout tests (quixbugs Cluster 2,
humaneval C3) by loosening unwinding — `--no-unwinding-assertions` paired with a
truncated loop produces false `SUCCESSFUL` and is banned by project policy. Use
`--k-induction` with required convergence instead, or raise the bound only when
the loop provably terminates within it.
